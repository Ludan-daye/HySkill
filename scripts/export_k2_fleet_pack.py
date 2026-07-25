#!/usr/bin/env python3
"""Export the strictly validated five-file K=2 fleet public pack."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    JsonLike,
    JsonObject,
    JsonValue,
    canonical_json,
    sha256_file,
)


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
DOMAIN_HELDOUT_COUNTS: dict[str, int] = {
    domain: DOMAIN_COUNTS[domain] - DOMAIN_VALIDATION_COUNTS[domain]
    for domain in DOMAINS
}
SEVEN_MODELS: tuple[str, ...] = (
    "deepseek7b",
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
    "yi15-9b",
)
FIVE_MODELS: tuple[str, ...] = (
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
)
FIXED_MODEL: str = "qwen3.5-4b-reference"
SELECT_UNAVAILABLE_MODELS: tuple[str, ...] = ("deepseek7b", "yi15-9b")
SPLITS: tuple[str, ...] = ("full", "heldout")
LOADING_ARMS: tuple[str, ...] = (
    "routed_always",
    "routed_gated",
    "routed_select",
)
ANSWER_ARMS: tuple[str, ...] = (
    "routed_always",
    "routed_gated",
    "routed_select",
    "fixed_gated",
)
EXPECTED_TOTAL_PER_MODEL: int = sum(DOMAIN_COUNTS.values())
EXPECTED_HELDOUT_PER_MODEL: int = sum(DOMAIN_HELDOUT_COUNTS.values())
EXPECTED_LOADING_DECISIONS: int = 53_770
EXPECTED_METRIC_ROWS: int = 210
EXPECTED_LOADING_INPUTS: int = 48
EXPECTED_ANSWER_INPUTS: int = 80
BOOTSTRAP_SAMPLES: int = 10_000
BOOTSTRAP_SEED: int = 0
SHA256_LENGTH: int = 64
GZIP_OPERATING_SYSTEM_OFFSET: int = 9
GZIP_OPERATING_SYSTEM_UNIX: int = 3
PUBLIC_SUMMARY_SCHEMA: str = "k2-fleet-summary-v1"
PUBLIC_MANIFEST_SCHEMA: str = "k2-fleet-public-pack-v1"
OUTPUT_FILENAMES: tuple[str, ...] = (
    "loading_metrics_long.jsonl.gz",
    "answer_metrics_long.jsonl.gz",
    "summary.json",
    "paired_comparisons.json",
    "manifest.json",
)
PAIRED_NOTES: str = (
    "Full support is descriptive. All paired inference excludes the frozen "
    "calibration IDs. Fleet intervals resample models, then domains, then "
    "paired instances."
)

MetricKey = tuple[str, str, str | None, str | None, str, str | None]


class ComparisonSpec(TypedDict):
    """Frozen fields for one current K=2 comparison."""

    arm_a: str
    arm_b: str
    models: tuple[str, ...]


COMPARISON_SPECS: dict[str, ComparisonSpec] = {
    "gated_vs_always_seven_model": {
        "arm_a": "routed_gated",
        "arm_b": "routed_always",
        "models": SEVEN_MODELS,
    },
    "gated_vs_select_five_model": {
        "arm_a": "routed_gated",
        "arm_b": "routed_select",
        "models": FIVE_MODELS,
    },
    "select_vs_always_five_model": {
        "arm_a": "routed_select",
        "arm_b": "routed_always",
        "models": FIVE_MODELS,
    },
    "routed_gated_vs_fixed_gated": {
        "arm_a": "routed_gated",
        "arm_b": "fixed_gated",
        "models": (FIXED_MODEL,),
    },
}

LOADING_BASE_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "level",
        "split",
        "model",
        "domain",
        "arm",
        "support",
        "instances",
        "loaded",
        "gold_loaded",
        "method_failures",
        "loaded_skill_precision",
        "loading_rate",
        "gold_load_rate",
        "selection_failure_rate",
    }
)
LOADING_MACRO_FIELDS: frozenset[str] = frozenset(
    {
        *LOADING_BASE_FIELDS,
        "models",
        "metric_model_denominators",
    }
)
ANSWER_BASE_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "level",
        "split",
        "model",
        "domain",
        "arm",
        "support",
        "n",
        "correct",
        "accuracy",
        "failure_categories",
    }
)
ANSWER_MACRO_FIELDS: frozenset[str] = frozenset(
    {*ANSWER_BASE_FIELDS, "fleet_micro_accuracy"}
)
RATE_FIELDS: tuple[str, ...] = (
    "loaded_skill_precision",
    "loading_rate",
    "gold_load_rate",
    "selection_failure_rate",
)


def parse_args() -> argparse.Namespace:
    """Parse the five explicit aggregate inputs and immutable output path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--loading-metrics-long", required=True, type=Path)
    parser.add_argument("--loading-summary", required=True, type=Path)
    parser.add_argument("--answer-metrics-long", required=True, type=Path)
    parser.add_argument("--answer-summary", required=True, type=Path)
    parser.add_argument(
        "--paired-comparisons-current4",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def require_exact_keys(
    value: Mapping[str, JsonValue],
    expected: frozenset[str],
    context: str,
) -> None:
    """Reject missing and unknown object fields."""

    actual: set[str] = set(value)
    if actual != set(expected):
        raise DownstreamDataError(
            "JSON object fields do not match the frozen schema: "
            f"context={context}, missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return a JSON object or raise a contextual error."""

    if not isinstance(value, dict):
        raise DownstreamDataError(
            f"Expected JSON object: context={context}, "
            f"actual={type(value).__name__}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return a JSON list or raise a contextual error."""

    if not isinstance(value, list):
        raise DownstreamDataError(
            f"Expected JSON list: context={context}, "
            f"actual={type(value).__name__}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return a non-empty JSON string."""

    if not isinstance(value, str) or not value:
        raise DownstreamDataError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_integer(value: JsonValue | None, context: str) -> int:
    """Return a JSON integer."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise DownstreamDataError(
            f"Expected integer: context={context}, value={value!r}"
        )
    return value


def require_number(value: JsonValue | None, context: str) -> float:
    """Return a finite JSON number."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise DownstreamDataError(
            f"Expected finite number: context={context}, value={value!r}"
        )
    return float(value)


def require_boolean(value: JsonValue | None, context: str) -> bool:
    """Return a JSON Boolean."""

    if not isinstance(value, bool):
        raise DownstreamDataError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


def require_string_list(
    value: JsonValue | None,
    context: str,
) -> tuple[str, ...]:
    """Return an ordered list of unique non-empty strings."""

    raw_values: list[JsonValue] = require_list(value, context)
    values: tuple[str, ...] = tuple(
        require_string(raw_value, f"{context}[{index}]")
        for index, raw_value in enumerate(raw_values)
    )
    if len(values) != len(set(values)):
        raise DownstreamDataError(
            f"String list contains duplicates: context={context}, values={values}"
        )
    return values


def require_sha256(value: JsonValue | None, context: str) -> str:
    """Return one lowercase SHA-256 digest."""

    digest: str = require_string(value, context)
    if (
        len(digest) != SHA256_LENGTH
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise DownstreamDataError(
            f"Expected lowercase SHA-256: context={context}, value={digest!r}"
        )
    return digest


def require_null(value: JsonValue | None, context: str) -> None:
    """Require an explicit JSON null."""

    if value is not None:
        raise DownstreamDataError(
            f"Expected null: context={context}, value={value!r}"
        )


def require_rate(
    value: JsonValue | None,
    context: str,
    allow_null: bool,
) -> float | None:
    """Return one probability, optionally accepting null."""

    if value is None:
        if allow_null:
            return None
        raise DownstreamDataError(
            f"Probability cannot be null: context={context}"
        )
    number: float = require_number(value, context)
    if number < 0.0 or number > 1.0:
        raise DownstreamDataError(
            f"Probability is outside [0, 1]: context={context}, value={number}"
        )
    return number


def assert_close(actual: float, expected: float, context: str) -> None:
    """Require deterministic floating-point agreement."""

    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise DownstreamDataError(
            f"Numeric value mismatch: context={context}, "
            f"expected={expected}, actual={actual}"
        )


def load_json_object(path: Path, context: str) -> JsonObject:
    """Load one strict JSON object."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required fleet input does not exist: context={context}, path={path}"
        )
    try:
        value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise DownstreamDataError(
            "Fleet JSON input is malformed: "
            f"context={context}, path={path}, line={error.lineno}, "
            f"column={error.colno}, message={error.msg}"
        ) from error
    return require_object(value, context)


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load a JSONL file while rejecting blank and non-object records."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required fleet input does not exist: context={context}, path={path}"
        )
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise DownstreamDataError(
                    "Fleet JSONL contains a blank record: "
                    f"context={context}, path={path}, line={line_number}"
                )
            try:
                value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise DownstreamDataError(
                    "Fleet JSONL input is malformed: "
                    f"context={context}, path={path}, line={line_number}, "
                    f"column={error.colno}, message={error.msg}"
                ) from error
            rows.append(
                require_object(value, f"{context}:{path}:{line_number}")
            )
    if len(rows) != EXPECTED_METRIC_ROWS:
        raise DownstreamDataError(
            "Fleet metric row count mismatch: "
            f"context={context}, expected={EXPECTED_METRIC_ROWS}, "
            f"actual={len(rows)}"
        )
    return rows


def loading_arms_for_model(model: str) -> tuple[str, ...]:
    """Return the frozen loading arms for one model."""

    if model in FIVE_MODELS:
        return LOADING_ARMS
    return ("routed_always", "routed_gated")


def answer_arms_for_model(model: str) -> tuple[str, ...]:
    """Return the frozen answer arms for one model."""

    arms: list[str] = ["routed_always", "routed_gated"]
    if model in FIVE_MODELS:
        arms.append("routed_select")
    if model == FIXED_MODEL:
        arms.append("fixed_gated")
    return tuple(arms)


def metric_key(row: JsonObject, context: str) -> MetricKey:
    """Return the unique coordinate of one long-form metric."""

    raw_model: JsonValue | None = row.get("model")
    model: str | None = (
        None if raw_model is None else require_string(raw_model, f"{context}.model")
    )
    raw_domain: JsonValue | None = row.get("domain")
    domain: str | None = (
        None
        if raw_domain is None
        else require_string(raw_domain, f"{context}.domain")
    )
    raw_support: JsonValue | None = row.get("support")
    support: str | None = (
        None
        if raw_support is None
        else require_string(raw_support, f"{context}.support")
    )
    return (
        require_string(row.get("level"), f"{context}.level"),
        require_string(row.get("split"), f"{context}.split"),
        model,
        domain,
        require_string(row.get("arm"), f"{context}.arm"),
        support,
    )


def expected_loading_keys() -> set[MetricKey]:
    """Return the exact 210-row loading metric coordinate set."""

    keys: set[MetricKey] = set()
    for split in SPLITS:
        for model in SEVEN_MODELS:
            for arm in loading_arms_for_model(model):
                keys.add(("per_model_pooled", split, model, None, arm, None))
                for domain in DOMAINS:
                    keys.add(
                        (
                            "per_model_domain",
                            split,
                            model,
                            domain,
                            arm,
                            None,
                        )
                    )
        for support, arms in (
            ("seven_model", ("routed_always", "routed_gated")),
            ("five_model", LOADING_ARMS),
        ):
            for arm in arms:
                keys.add(("fleet_micro", split, None, None, arm, support))
                keys.add(
                    ("fleet_model_macro", split, None, None, arm, support)
                )
    return keys


def expected_answer_keys() -> set[MetricKey]:
    """Return the exact 210-row answer metric coordinate set."""

    keys: set[MetricKey] = set()
    for split in SPLITS:
        for model in SEVEN_MODELS:
            for arm in answer_arms_for_model(model):
                keys.add(("model_pooled", split, model, None, arm, None))
                for domain in DOMAINS:
                    keys.add(
                        ("model_domain", split, model, domain, arm, None)
                    )
        for support, arms in (
            ("seven_model", ("routed_always", "routed_gated")),
            ("five_model", LOADING_ARMS),
        ):
            for arm in arms:
                keys.add(
                    ("fleet_model_macro", split, None, None, arm, support)
                )
    return keys


def expected_instances(
    split: str,
    domain: str | None,
    model_count: int,
) -> int:
    """Return the frozen denominator for one metric coordinate."""

    per_model: int
    if domain is None:
        per_model = (
            EXPECTED_TOTAL_PER_MODEL
            if split == "full"
            else EXPECTED_HELDOUT_PER_MODEL
        )
    else:
        per_model = (
            DOMAIN_COUNTS[domain]
            if split == "full"
            else DOMAIN_HELDOUT_COUNTS[domain]
        )
    return per_model * model_count


def validate_loading_row(row: JsonObject, context: str) -> None:
    """Validate one loading metric row without trusting its producer."""

    level: str = require_string(row.get("level"), f"{context}.level")
    expected_fields: frozenset[str] = (
        LOADING_MACRO_FIELDS
        if level == "fleet_model_macro"
        else LOADING_BASE_FIELDS
    )
    require_exact_keys(row, expected_fields, context)
    if row.get("schema_version") != "k2-loading-metrics-v1":
        raise DownstreamDataError(
            f"Unexpected loading schema: context={context}, "
            f"schema={row.get('schema_version')!r}"
        )
    if level not in (
        "per_model_domain",
        "per_model_pooled",
        "fleet_micro",
        "fleet_model_macro",
    ):
        raise DownstreamDataError(
            f"Unknown loading level: context={context}, level={level}"
        )
    split: str = require_string(row.get("split"), f"{context}.split")
    if split not in SPLITS:
        raise DownstreamDataError(
            f"Unknown loading split: context={context}, split={split}"
        )
    arm: str = require_string(row.get("arm"), f"{context}.arm")
    if arm not in LOADING_ARMS:
        raise DownstreamDataError(
            f"Unknown loading arm: context={context}, arm={arm}"
        )
    raw_model: JsonValue | None = row.get("model")
    raw_domain: JsonValue | None = row.get("domain")
    raw_support: JsonValue | None = row.get("support")
    model_count: int = 1
    domain: str | None = None
    if level.startswith("per_model"):
        model: str = require_string(raw_model, f"{context}.model")
        if model not in SEVEN_MODELS or arm not in loading_arms_for_model(model):
            raise DownstreamDataError(
                f"Loading model-arm support mismatch: context={context}, "
                f"model={model}, arm={arm}"
            )
        require_null(raw_support, f"{context}.support")
        if level == "per_model_domain":
            domain = require_string(raw_domain, f"{context}.domain")
            if domain not in DOMAINS:
                raise DownstreamDataError(
                    f"Unknown loading domain: context={context}, domain={domain}"
                )
        else:
            require_null(raw_domain, f"{context}.domain")
    else:
        require_null(raw_model, f"{context}.model")
        require_null(raw_domain, f"{context}.domain")
        support: str = require_string(raw_support, f"{context}.support")
        if support not in ("seven_model", "five_model"):
            raise DownstreamDataError(
                f"Unknown loading support: context={context}, support={support}"
            )
        model_count = 7 if support == "seven_model" else 5
        allowed_arms: tuple[str, ...] = (
            ("routed_always", "routed_gated")
            if support == "seven_model"
            else LOADING_ARMS
        )
        if arm not in allowed_arms:
            raise DownstreamDataError(
                f"Loading fleet arm support mismatch: context={context}, "
                f"support={support}, arm={arm}"
            )
    instances: int = require_integer(
        row.get("instances"),
        f"{context}.instances",
    )
    expected_count: int = expected_instances(split, domain, model_count)
    if instances != expected_count:
        raise DownstreamDataError(
            f"Loading denominator mismatch: context={context}, "
            f"expected={expected_count}, actual={instances}"
        )
    loaded: int = require_integer(row.get("loaded"), f"{context}.loaded")
    gold_loaded: int = require_integer(
        row.get("gold_loaded"),
        f"{context}.gold_loaded",
    )
    method_failures: int = require_integer(
        row.get("method_failures"),
        f"{context}.method_failures",
    )
    if not (
        0 <= gold_loaded <= loaded <= instances
        and 0 <= method_failures <= instances
    ):
        raise DownstreamDataError(
            "Loading counts are inconsistent: "
            f"context={context}, instances={instances}, loaded={loaded}, "
            f"gold_loaded={gold_loaded}, method_failures={method_failures}"
        )
    precision: float | None = require_rate(
        row.get("loaded_skill_precision"),
        f"{context}.loaded_skill_precision",
        True,
    )
    loading_rate: float = cast(
        float,
        require_rate(row.get("loading_rate"), f"{context}.loading_rate", False),
    )
    gold_rate: float = cast(
        float,
        require_rate(
            row.get("gold_load_rate"),
            f"{context}.gold_load_rate",
            False,
        ),
    )
    failure_rate: float = cast(
        float,
        require_rate(
            row.get("selection_failure_rate"),
            f"{context}.selection_failure_rate",
            False,
        ),
    )
    if level != "fleet_model_macro":
        expected_precision: float | None = (
            None if loaded == 0 else gold_loaded / loaded
        )
        if precision is None or expected_precision is None:
            if precision != expected_precision:
                raise DownstreamDataError(
                    f"Loading precision nullability mismatch: context={context}"
                )
        else:
            assert_close(
                precision,
                expected_precision,
                f"{context}.loaded_skill_precision",
            )
        assert_close(loading_rate, loaded / instances, f"{context}.loading_rate")
        assert_close(gold_rate, gold_loaded / instances, f"{context}.gold_load_rate")
        assert_close(
            failure_rate,
            method_failures / instances,
            f"{context}.selection_failure_rate",
        )
    else:
        models: int = require_integer(row.get("models"), f"{context}.models")
        if models != model_count:
            raise DownstreamDataError(
                f"Loading macro model count mismatch: context={context}, "
                f"expected={model_count}, actual={models}"
            )
        denominators: JsonObject = require_object(
            row.get("metric_model_denominators"),
            f"{context}.metric_model_denominators",
        )
        require_exact_keys(
            denominators,
            frozenset(RATE_FIELDS),
            f"{context}.metric_model_denominators",
        )
        for field in RATE_FIELDS:
            denominator: int = require_integer(
                denominators.get(field),
                f"{context}.metric_model_denominators.{field}",
            )
            if denominator < 0 or denominator > model_count:
                raise DownstreamDataError(
                    "Loading macro metric denominator is invalid: "
                    f"context={context}, field={field}, value={denominator}"
                )


def loading_counts(row: JsonObject) -> tuple[int, int, int, int]:
    """Return the four additive loading counts."""

    return (
        require_integer(row.get("instances"), "loading.instances"),
        require_integer(row.get("loaded"), "loading.loaded"),
        require_integer(row.get("gold_loaded"), "loading.gold_loaded"),
        require_integer(row.get("method_failures"), "loading.method_failures"),
    )


def validate_loading_aggregate(
    aggregate: JsonObject,
    children: Sequence[JsonObject],
    macro: bool,
    context: str,
) -> None:
    """Recompute one pooled, micro, or model-macro loading row."""

    if not children:
        raise DownstreamDataError(
            f"Loading aggregate has no children: context={context}"
        )
    child_counts: list[tuple[int, int, int, int]] = [
        loading_counts(child) for child in children
    ]
    expected_counts: tuple[int, int, int, int] = tuple(
        sum(count[index] for count in child_counts)
        for index in range(4)
    )
    actual_counts: tuple[int, int, int, int] = loading_counts(aggregate)
    if actual_counts != expected_counts:
        raise DownstreamDataError(
            f"Loading aggregate counts mismatch: context={context}, "
            f"expected={expected_counts}, actual={actual_counts}"
        )
    instances, loaded, gold_loaded, failures = expected_counts
    expected_rates: dict[str, float | None]
    if macro:
        expected_rates = {}
        for field in RATE_FIELDS:
            values: list[float] = [
                require_number(child.get(field), f"{context}.{field}")
                for child in children
                if child.get(field) is not None
            ]
            expected_rates[field] = (
                None if not values else sum(values) / len(values)
            )
            denominators: JsonObject = require_object(
                aggregate.get("metric_model_denominators"),
                f"{context}.metric_model_denominators",
            )
            actual_denominator: int = require_integer(
                denominators.get(field),
                f"{context}.metric_model_denominators.{field}",
            )
            if actual_denominator != len(values):
                raise DownstreamDataError(
                    "Loading macro metric denominator mismatch: "
                    f"context={context}, field={field}, "
                    f"expected={len(values)}, actual={actual_denominator}"
                )
    else:
        expected_rates = {
            "loaded_skill_precision": (
                None if loaded == 0 else gold_loaded / loaded
            ),
            "loading_rate": loaded / instances,
            "gold_load_rate": gold_loaded / instances,
            "selection_failure_rate": failures / instances,
        }
    for field, expected_rate in expected_rates.items():
        actual_value: JsonValue | None = aggregate.get(field)
        if expected_rate is None:
            require_null(actual_value, f"{context}.{field}")
        else:
            assert_close(
                require_number(actual_value, f"{context}.{field}"),
                expected_rate,
                f"{context}.{field}",
            )


def validate_loading_metrics(rows: Sequence[JsonObject]) -> dict[MetricKey, JsonObject]:
    """Validate all 210 loading rows and their aggregate relationships."""

    index: dict[MetricKey, JsonObject] = {}
    for row_index, row in enumerate(rows):
        context: str = f"loading-metrics[{row_index}]"
        validate_loading_row(row, context)
        key: MetricKey = metric_key(row, context)
        if key in index:
            raise DownstreamDataError(
                f"Duplicate loading metric coordinate: key={key}"
            )
        index[key] = row
    expected: set[MetricKey] = expected_loading_keys()
    if set(index) != expected:
        raise DownstreamDataError(
            "Loading metric support mismatch: "
            f"missing={sorted(expected - set(index), key=str)[:10]}, "
            f"unexpected={sorted(set(index) - expected, key=str)[:10]}"
        )
    for split in SPLITS:
        for model in SEVEN_MODELS:
            for arm in loading_arms_for_model(model):
                pooled_key: MetricKey = (
                    "per_model_pooled",
                    split,
                    model,
                    None,
                    arm,
                    None,
                )
                domain_rows: list[JsonObject] = [
                    index[
                        (
                            "per_model_domain",
                            split,
                            model,
                            domain,
                            arm,
                            None,
                        )
                    ]
                    for domain in DOMAINS
                ]
                validate_loading_aggregate(
                    index[pooled_key],
                    domain_rows,
                    False,
                    f"loading-pooled:{split}:{model}:{arm}",
                )
        for support, models, arms in (
            ("seven_model", SEVEN_MODELS, ("routed_always", "routed_gated")),
            ("five_model", FIVE_MODELS, LOADING_ARMS),
        ):
            for arm in arms:
                model_rows: list[JsonObject] = [
                    index[
                        (
                            "per_model_pooled",
                            split,
                            model,
                            None,
                            arm,
                            None,
                        )
                    ]
                    for model in models
                ]
                validate_loading_aggregate(
                    index[
                        ("fleet_micro", split, None, None, arm, support)
                    ],
                    model_rows,
                    False,
                    f"loading-micro:{split}:{support}:{arm}",
                )
                validate_loading_aggregate(
                    index[
                        (
                            "fleet_model_macro",
                            split,
                            None,
                            None,
                            arm,
                            support,
                        )
                    ],
                    model_rows,
                    True,
                    f"loading-macro:{split}:{support}:{arm}",
                )
    return index


def validate_failure_categories(
    value: JsonValue | None,
    expected_total: int,
    context: str,
) -> None:
    """Validate the published answer outcome accounting."""

    categories: JsonObject = require_object(value, context)
    allowed: set[str] = {"success", "method_failure"}
    unexpected: set[str] = set(categories) - allowed
    if unexpected or "success" not in categories:
        raise DownstreamDataError(
            f"Unknown or missing answer failure categories: context={context}, "
            f"unexpected={sorted(unexpected)}, keys={sorted(categories)}"
        )
    total: int = 0
    for category, raw_count in categories.items():
        count: int = require_integer(raw_count, f"{context}.{category}")
        if count < 0:
            raise DownstreamDataError(
                f"Negative answer outcome count: context={context}.{category}"
            )
        total += count
    if total != expected_total:
        raise DownstreamDataError(
            f"Answer outcome count mismatch: context={context}, "
            f"expected={expected_total}, actual={total}"
        )


def validate_answer_row(row: JsonObject, context: str) -> None:
    """Validate one answer metric row."""

    level: str = require_string(row.get("level"), f"{context}.level")
    expected_fields: frozenset[str] = (
        ANSWER_MACRO_FIELDS
        if level == "fleet_model_macro"
        else ANSWER_BASE_FIELDS
    )
    require_exact_keys(row, expected_fields, context)
    if row.get("schema_version") != "k2-answer-metrics-long-v1":
        raise DownstreamDataError(
            f"Unexpected answer metric schema: context={context}, "
            f"schema={row.get('schema_version')!r}"
        )
    if level not in ("model_domain", "model_pooled", "fleet_model_macro"):
        raise DownstreamDataError(
            f"Unknown answer metric level: context={context}, level={level}"
        )
    split: str = require_string(row.get("split"), f"{context}.split")
    if split not in SPLITS:
        raise DownstreamDataError(
            f"Unknown answer split: context={context}, split={split}"
        )
    arm: str = require_string(row.get("arm"), f"{context}.arm")
    if arm not in ANSWER_ARMS:
        raise DownstreamDataError(
            f"Unknown answer arm: context={context}, arm={arm}"
        )
    raw_model: JsonValue | None = row.get("model")
    raw_domain: JsonValue | None = row.get("domain")
    raw_support: JsonValue | None = row.get("support")
    model_count: int = 1
    domain: str | None = None
    if level.startswith("model_"):
        model: str = require_string(raw_model, f"{context}.model")
        if model not in SEVEN_MODELS or arm not in answer_arms_for_model(model):
            raise DownstreamDataError(
                f"Answer model-arm support mismatch: context={context}, "
                f"model={model}, arm={arm}"
            )
        require_null(raw_support, f"{context}.support")
        if level == "model_domain":
            domain = require_string(raw_domain, f"{context}.domain")
            if domain not in DOMAINS:
                raise DownstreamDataError(
                    f"Unknown answer domain: context={context}, domain={domain}"
                )
        else:
            require_null(raw_domain, f"{context}.domain")
    else:
        require_null(raw_model, f"{context}.model")
        require_null(raw_domain, f"{context}.domain")
        support: str = require_string(raw_support, f"{context}.support")
        if support not in ("seven_model", "five_model"):
            raise DownstreamDataError(
                f"Unknown answer support: context={context}, support={support}"
            )
        model_count = 7 if support == "seven_model" else 5
        allowed_arms: tuple[str, ...] = (
            ("routed_always", "routed_gated")
            if support == "seven_model"
            else LOADING_ARMS
        )
        if arm not in allowed_arms:
            raise DownstreamDataError(
                f"Answer fleet arm support mismatch: context={context}, "
                f"support={support}, arm={arm}"
            )
    total: int = require_integer(row.get("n"), f"{context}.n")
    expected_total: int = expected_instances(split, domain, model_count)
    if total != expected_total:
        raise DownstreamDataError(
            f"Answer denominator mismatch: context={context}, "
            f"expected={expected_total}, actual={total}"
        )
    correct: int = require_integer(row.get("correct"), f"{context}.correct")
    if correct < 0 or correct > total:
        raise DownstreamDataError(
            f"Answer correct count is invalid: context={context}, "
            f"correct={correct}, total={total}"
        )
    accuracy: float = cast(
        float,
        require_rate(row.get("accuracy"), f"{context}.accuracy", False),
    )
    validate_failure_categories(
        row.get("failure_categories"),
        total,
        f"{context}.failure_categories",
    )
    if level != "fleet_model_macro":
        assert_close(accuracy, correct / total, f"{context}.accuracy")
    else:
        micro_accuracy: float = cast(
            float,
            require_rate(
                row.get("fleet_micro_accuracy"),
                f"{context}.fleet_micro_accuracy",
                False,
            ),
        )
        assert_close(
            micro_accuracy,
            correct / total,
            f"{context}.fleet_micro_accuracy",
        )


def merge_categories(rows: Sequence[JsonObject], context: str) -> dict[str, int]:
    """Sum answer outcome categories across child rows."""

    categories: dict[str, int] = {}
    for row in rows:
        child: JsonObject = require_object(
            row.get("failure_categories"),
            f"{context}.failure_categories",
        )
        for category, raw_count in child.items():
            count: int = require_integer(
                raw_count,
                f"{context}.failure_categories.{category}",
            )
            categories[category] = categories.get(category, 0) + count
    return categories


def validate_answer_aggregate(
    aggregate: JsonObject,
    children: Sequence[JsonObject],
    macro: bool,
    context: str,
) -> None:
    """Recompute one pooled or model-macro answer metric."""

    expected_n: int = sum(
        require_integer(child.get("n"), f"{context}.child.n")
        for child in children
    )
    expected_correct: int = sum(
        require_integer(child.get("correct"), f"{context}.child.correct")
        for child in children
    )
    actual_n: int = require_integer(aggregate.get("n"), f"{context}.n")
    actual_correct: int = require_integer(
        aggregate.get("correct"),
        f"{context}.correct",
    )
    if (actual_n, actual_correct) != (expected_n, expected_correct):
        raise DownstreamDataError(
            f"Answer aggregate counts mismatch: context={context}, "
            f"expected={(expected_n, expected_correct)}, "
            f"actual={(actual_n, actual_correct)}"
        )
    expected_categories: dict[str, int] = merge_categories(children, context)
    actual_categories: JsonObject = require_object(
        aggregate.get("failure_categories"),
        f"{context}.failure_categories",
    )
    if canonical_json(actual_categories) != canonical_json(expected_categories):
        raise DownstreamDataError(
            f"Answer aggregate categories mismatch: context={context}"
        )
    expected_accuracy: float = (
        sum(
            require_number(
                child.get("accuracy"),
                f"{context}.child.accuracy",
            )
            for child in children
        )
        / len(children)
        if macro
        else expected_correct / expected_n
    )
    assert_close(
        require_number(aggregate.get("accuracy"), f"{context}.accuracy"),
        expected_accuracy,
        f"{context}.accuracy",
    )
    if macro:
        assert_close(
            require_number(
                aggregate.get("fleet_micro_accuracy"),
                f"{context}.fleet_micro_accuracy",
            ),
            expected_correct / expected_n,
            f"{context}.fleet_micro_accuracy",
        )


def validate_answer_metrics(rows: Sequence[JsonObject]) -> dict[MetricKey, JsonObject]:
    """Validate all 210 answer rows and their aggregate relationships."""

    index: dict[MetricKey, JsonObject] = {}
    for row_index, row in enumerate(rows):
        context: str = f"answer-metrics[{row_index}]"
        validate_answer_row(row, context)
        key: MetricKey = metric_key(row, context)
        if key in index:
            raise DownstreamDataError(
                f"Duplicate answer metric coordinate: key={key}"
            )
        index[key] = row
    expected: set[MetricKey] = expected_answer_keys()
    if set(index) != expected:
        raise DownstreamDataError(
            "Answer metric support mismatch: "
            f"missing={sorted(expected - set(index), key=str)[:10]}, "
            f"unexpected={sorted(set(index) - expected, key=str)[:10]}"
        )
    for split in SPLITS:
        for model in SEVEN_MODELS:
            for arm in answer_arms_for_model(model):
                domain_rows: list[JsonObject] = [
                    index[("model_domain", split, model, domain, arm, None)]
                    for domain in DOMAINS
                ]
                validate_answer_aggregate(
                    index[("model_pooled", split, model, None, arm, None)],
                    domain_rows,
                    False,
                    f"answer-pooled:{split}:{model}:{arm}",
                )
        for support, models, arms in (
            ("seven_model", SEVEN_MODELS, ("routed_always", "routed_gated")),
            ("five_model", FIVE_MODELS, LOADING_ARMS),
        ):
            for arm in arms:
                model_rows: list[JsonObject] = [
                    index[("model_pooled", split, model, None, arm, None)]
                    for model in models
                ]
                validate_answer_aggregate(
                    index[
                        (
                            "fleet_model_macro",
                            split,
                            None,
                            None,
                            arm,
                            support,
                        )
                    ],
                    model_rows,
                    True,
                    f"answer-macro:{split}:{support}:{arm}",
                )
    return index


def validate_source_entries(
    value: JsonValue | None,
    expected_count: int,
    context: str,
) -> None:
    """Validate source identities without dereferencing private paths."""

    entries: list[JsonValue] = require_list(value, context)
    if len(entries) != expected_count:
        raise DownstreamDataError(
            f"Source file count mismatch: context={context}, "
            f"expected={expected_count}, actual={len(entries)}"
        )
    paths: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry_context: str = f"{context}[{index}]"
        entry: JsonObject = require_object(raw_entry, entry_context)
        require_exact_keys(
            entry,
            frozenset({"path", "sha256"}),
            entry_context,
        )
        path: str = require_string(entry.get("path"), f"{entry_context}.path")
        require_sha256(entry.get("sha256"), f"{entry_context}.sha256")
        if path in paths:
            raise DownstreamDataError(
                f"Duplicate source path: context={context}, path={path}"
            )
        paths.add(path)


def loading_fleet_rows(
    index: Mapping[MetricKey, JsonObject],
) -> list[JsonObject]:
    """Return loading model-macro rows in deterministic protocol order."""

    rows: list[JsonObject] = []
    for split in SPLITS:
        for support, arms in (
            ("seven_model", ("routed_always", "routed_gated")),
            ("five_model", LOADING_ARMS),
        ):
            for arm in arms:
                rows.append(
                    index[
                        (
                            "fleet_model_macro",
                            split,
                            None,
                            None,
                            arm,
                            support,
                        )
                    ]
                )
    return rows


def answer_summary_blocks(
    index: Mapping[MetricKey, JsonObject],
) -> tuple[JsonObject, JsonObject]:
    """Rebuild the nested answer summary solely from validated long rows."""

    model_pooled: JsonObject = {}
    for model in SEVEN_MODELS:
        model_block: JsonObject = {}
        for split in SPLITS:
            split_block: JsonObject = {}
            for arm in answer_arms_for_model(model):
                split_block[arm] = index[
                    ("model_pooled", split, model, None, arm, None)
                ]
            model_block[split] = split_block
        model_pooled[model] = model_block
    fleet: JsonObject = {}
    for support, arms in (
        ("seven_model", ("routed_always", "routed_gated")),
        ("five_model", LOADING_ARMS),
    ):
        support_block: JsonObject = {}
        for split in SPLITS:
            split_block = {}
            for arm in arms:
                split_block[arm] = index[
                    (
                        "fleet_model_macro",
                        split,
                        None,
                        None,
                        arm,
                        support,
                    )
                ]
            support_block[split] = split_block
        fleet[support] = support_block
    return model_pooled, fleet


def validate_loading_summary(
    payload: JsonObject,
    metrics_path: Path,
    fleet_rows: Sequence[JsonObject],
) -> None:
    """Validate the private loading summary and its long-file binding."""

    require_exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "valid",
                "input_files",
                "input_records",
                "seven_model_support",
                "five_model_support",
                "expected_total_per_model",
                "expected_heldout_per_model",
                "fleet_model_macro",
                "long_metrics_path",
                "long_metrics_sha256",
            }
        ),
        "loading-summary",
    )
    if payload.get("schema_version") != "k2-loading-summary-v1":
        raise DownstreamDataError(
            "Unexpected loading summary schema: "
            f"schema={payload.get('schema_version')!r}"
        )
    if not require_boolean(payload.get("valid"), "loading-summary.valid"):
        raise DownstreamDataError("Loading summary is not valid")
    if require_integer(
        payload.get("input_records"),
        "loading-summary.input_records",
    ) != EXPECTED_LOADING_DECISIONS:
        raise DownstreamDataError(
            "Loading decision count does not match the frozen protocol"
        )
    if require_integer(
        payload.get("expected_total_per_model"),
        "loading-summary.expected_total_per_model",
    ) != EXPECTED_TOTAL_PER_MODEL:
        raise DownstreamDataError("Loading total denominator is invalid")
    if require_integer(
        payload.get("expected_heldout_per_model"),
        "loading-summary.expected_heldout_per_model",
    ) != EXPECTED_HELDOUT_PER_MODEL:
        raise DownstreamDataError("Loading held-out denominator is invalid")
    if require_string_list(
        payload.get("seven_model_support"),
        "loading-summary.seven_model_support",
    ) != SEVEN_MODELS:
        raise DownstreamDataError("Loading seven-model support is invalid")
    if require_string_list(
        payload.get("five_model_support"),
        "loading-summary.five_model_support",
    ) != FIVE_MODELS:
        raise DownstreamDataError("Loading five-model support is invalid")
    validate_source_entries(
        payload.get("input_files"),
        EXPECTED_LOADING_INPUTS,
        "loading-summary.input_files",
    )
    require_string(
        payload.get("long_metrics_path"),
        "loading-summary.long_metrics_path",
    )
    expected_sha256: str = sha256_file(metrics_path)
    actual_sha256: str = require_sha256(
        payload.get("long_metrics_sha256"),
        "loading-summary.long_metrics_sha256",
    )
    if actual_sha256 != expected_sha256:
        raise DownstreamDataError(
            "Loading long metrics hash mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    summary_rows: list[JsonValue] = require_list(
        payload.get("fleet_model_macro"),
        "loading-summary.fleet_model_macro",
    )
    if canonical_json(summary_rows) != canonical_json(list(fleet_rows)):
        raise DownstreamDataError(
            "Loading summary statistics disagree with validated long metrics"
        )


def validate_answer_summary(
    payload: JsonObject,
    model_pooled: JsonObject,
    fleet_model_macro: JsonObject,
) -> None:
    """Validate answer summary support and nested long-row statistics."""

    require_exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "model_pooled",
                "fleet_model_macro",
                "seven_models",
                "five_models",
                "fixed_model",
                "expected_total_per_model",
                "expected_heldout_per_model",
                "inputs",
            }
        ),
        "answer-summary",
    )
    if payload.get("schema_version") != "k2-answer-summary-v1":
        raise DownstreamDataError(
            "Unexpected answer summary schema: "
            f"schema={payload.get('schema_version')!r}"
        )
    if require_string_list(
        payload.get("seven_models"),
        "answer-summary.seven_models",
    ) != SEVEN_MODELS:
        raise DownstreamDataError("Answer seven-model support is invalid")
    if require_string_list(
        payload.get("five_models"),
        "answer-summary.five_models",
    ) != FIVE_MODELS:
        raise DownstreamDataError("Answer five-model support is invalid")
    if require_string(
        payload.get("fixed_model"),
        "answer-summary.fixed_model",
    ) != FIXED_MODEL:
        raise DownstreamDataError("Answer fixed-model support is invalid")
    if require_integer(
        payload.get("expected_total_per_model"),
        "answer-summary.expected_total_per_model",
    ) != EXPECTED_TOTAL_PER_MODEL:
        raise DownstreamDataError("Answer total denominator is invalid")
    if require_integer(
        payload.get("expected_heldout_per_model"),
        "answer-summary.expected_heldout_per_model",
    ) != EXPECTED_HELDOUT_PER_MODEL:
        raise DownstreamDataError("Answer held-out denominator is invalid")
    validate_source_entries(
        payload.get("inputs"),
        EXPECTED_ANSWER_INPUTS,
        "answer-summary.inputs",
    )
    raw_model_pooled: JsonObject = require_object(
        payload.get("model_pooled"),
        "answer-summary.model_pooled",
    )
    raw_fleet: JsonObject = require_object(
        payload.get("fleet_model_macro"),
        "answer-summary.fleet_model_macro",
    )
    if canonical_json(raw_model_pooled) != canonical_json(model_pooled):
        raise DownstreamDataError(
            "Answer model-pooled summary disagrees with validated long metrics"
        )
    if canonical_json(raw_fleet) != canonical_json(fleet_model_macro):
        raise DownstreamDataError(
            "Answer fleet summary disagrees with validated long metrics"
        )


def validate_ci(value: JsonValue | None, context: str) -> None:
    """Validate one ordered finite confidence interval."""

    raw_interval: list[JsonValue] = require_list(value, context)
    if len(raw_interval) != 2:
        raise DownstreamDataError(
            f"Confidence interval must contain two values: context={context}"
        )
    lower: float = require_number(raw_interval[0], f"{context}[0]")
    upper: float = require_number(raw_interval[1], f"{context}[1]")
    if lower > upper or lower < -1.0 or upper > 1.0:
        raise DownstreamDataError(
            f"Confidence interval is invalid: context={context}, "
            f"interval={[lower, upper]}"
        )


def answer_accuracy(
    index: Mapping[MetricKey, JsonObject],
    level: str,
    split: str,
    model: str,
    domain: str | None,
    arm: str,
) -> float:
    """Return one validated answer accuracy."""

    row: JsonObject = index[(level, split, model, domain, arm, None)]
    return require_number(row.get("accuracy"), "answer-accuracy")


def validate_bootstrap_stats(
    value: JsonValue | None,
    expected_n: int,
    expected_a: float,
    expected_b: float,
    context: str,
) -> None:
    """Validate one model-domain or pooled paired bootstrap record."""

    payload: JsonObject = require_object(value, context)
    require_exact_keys(
        payload,
        frozenset(
            {
                "n",
                "mean_k",
                "mean_k4",
                "difference_k_minus_k4",
                "ci95",
                "p_two_sided",
                "bootstrap_samples",
                "seed",
            }
        ),
        context,
    )
    if require_integer(payload.get("n"), f"{context}.n") != expected_n:
        raise DownstreamDataError(
            f"Paired denominator mismatch: context={context}, expected={expected_n}"
        )
    mean_a: float = require_number(payload.get("mean_k"), f"{context}.mean_k")
    mean_b: float = require_number(payload.get("mean_k4"), f"{context}.mean_k4")
    difference: float = require_number(
        payload.get("difference_k_minus_k4"),
        f"{context}.difference_k_minus_k4",
    )
    assert_close(mean_a, expected_a, f"{context}.mean_k")
    assert_close(mean_b, expected_b, f"{context}.mean_k4")
    assert_close(difference, mean_a - mean_b, f"{context}.difference")
    validate_ci(payload.get("ci95"), f"{context}.ci95")
    require_rate(payload.get("p_two_sided"), f"{context}.p_two_sided", False)
    if require_integer(
        payload.get("bootstrap_samples"),
        f"{context}.bootstrap_samples",
    ) != BOOTSTRAP_SAMPLES:
        raise DownstreamDataError(
            f"Paired bootstrap count mismatch: context={context}"
        )
    if require_integer(payload.get("seed"), f"{context}.seed") < 0:
        raise DownstreamDataError(
            f"Paired bootstrap seed is negative: context={context}"
        )


def validate_hierarchical(
    value: JsonValue | None,
    specification: ComparisonSpec,
    by_model: JsonObject,
    context: str,
) -> None:
    """Validate one fleet hierarchical-bootstrap report."""

    payload: JsonObject = require_object(value, context)
    require_exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "arm_a",
                "arm_b",
                "n_models",
                "n_domains_per_model",
                "n_unique_instances",
                "mean_a",
                "mean_b",
                "difference_a_minus_b",
                "ci95",
                "p_two_sided",
                "bootstrap_samples",
                "seed",
                "resampling_unit",
            }
        ),
        context,
    )
    if payload.get("schema_version") != "k2-answer-hierarchical-bootstrap-v1":
        raise DownstreamDataError(
            f"Unexpected hierarchical schema: context={context}"
        )
    if (
        payload.get("arm_a") != specification["arm_a"]
        or payload.get("arm_b") != specification["arm_b"]
    ):
        raise DownstreamDataError(
            f"Hierarchical arm mismatch: context={context}"
        )
    model_count: int = len(specification["models"])
    if require_integer(payload.get("n_models"), f"{context}.n_models") != model_count:
        raise DownstreamDataError(
            f"Hierarchical model count mismatch: context={context}"
        )
    if require_integer(
        payload.get("n_domains_per_model"),
        f"{context}.n_domains_per_model",
    ) != len(DOMAINS):
        raise DownstreamDataError(
            f"Hierarchical domain count mismatch: context={context}"
        )
    expected_instances: int = model_count * EXPECTED_HELDOUT_PER_MODEL
    if require_integer(
        payload.get("n_unique_instances"),
        f"{context}.n_unique_instances",
    ) != expected_instances:
        raise DownstreamDataError(
            f"Hierarchical instance count mismatch: context={context}"
        )
    means_a: list[float] = []
    means_b: list[float] = []
    for model in specification["models"]:
        model_block: JsonObject = require_object(
            by_model.get(model),
            f"{context}.by_model.{model}",
        )
        domains: JsonObject = require_object(
            model_block.get("by_domain"),
            f"{context}.by_model.{model}.by_domain",
        )
        for domain in DOMAINS:
            stats: JsonObject = require_object(
                domains.get(domain),
                f"{context}.by_model.{model}.by_domain.{domain}",
            )
            means_a.append(
                require_number(
                    stats.get("mean_k"),
                    f"{context}.by_model.{model}.{domain}.mean_k",
                )
            )
            means_b.append(
                require_number(
                    stats.get("mean_k4"),
                    f"{context}.by_model.{model}.{domain}.mean_k4",
                )
            )
    expected_a: float = sum(means_a) / len(means_a)
    expected_b: float = sum(means_b) / len(means_b)
    mean_a: float = require_number(payload.get("mean_a"), f"{context}.mean_a")
    mean_b: float = require_number(payload.get("mean_b"), f"{context}.mean_b")
    difference: float = require_number(
        payload.get("difference_a_minus_b"),
        f"{context}.difference_a_minus_b",
    )
    assert_close(mean_a, expected_a, f"{context}.mean_a")
    assert_close(mean_b, expected_b, f"{context}.mean_b")
    assert_close(difference, mean_a - mean_b, f"{context}.difference")
    validate_ci(payload.get("ci95"), f"{context}.ci95")
    require_rate(payload.get("p_two_sided"), f"{context}.p_two_sided", False)
    if require_integer(
        payload.get("bootstrap_samples"),
        f"{context}.bootstrap_samples",
    ) != BOOTSTRAP_SAMPLES:
        raise DownstreamDataError(
            f"Hierarchical bootstrap count mismatch: context={context}"
        )
    if require_integer(payload.get("seed"), f"{context}.seed") != BOOTSTRAP_SEED:
        raise DownstreamDataError(
            f"Hierarchical seed mismatch: context={context}"
        )
    if payload.get("resampling_unit") != (
        "models, then domains, then paired instances"
    ):
        raise DownstreamDataError(
            f"Hierarchical resampling unit mismatch: context={context}"
        )


def validate_comparisons(
    payload: JsonObject,
    answer_index: Mapping[MetricKey, JsonObject],
) -> None:
    """Validate exactly the four current K=2 paired comparisons."""

    require_exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "split",
                "bootstrap_samples",
                "bootstrap_seed",
                "comparisons",
                "notes",
            }
        ),
        "paired-comparisons",
    )
    if payload.get("schema_version") != "k2-answer-paired-comparisons-v1":
        raise DownstreamDataError("Unexpected paired-comparisons schema")
    if payload.get("split") != "heldout":
        raise DownstreamDataError("Paired comparisons must use heldout split")
    if require_integer(
        payload.get("bootstrap_samples"),
        "paired-comparisons.bootstrap_samples",
    ) != BOOTSTRAP_SAMPLES:
        raise DownstreamDataError("Paired bootstrap count is invalid")
    if require_integer(
        payload.get("bootstrap_seed"),
        "paired-comparisons.bootstrap_seed",
    ) != BOOTSTRAP_SEED:
        raise DownstreamDataError("Paired bootstrap seed is invalid")
    if payload.get("notes") != PAIRED_NOTES:
        raise DownstreamDataError("Paired-comparisons protocol note is invalid")
    comparisons: JsonObject = require_object(
        payload.get("comparisons"),
        "paired-comparisons.comparisons",
    )
    if set(comparisons) != set(COMPARISON_SPECS):
        raise DownstreamDataError(
            "Paired comparison set is not current4: "
            f"missing={sorted(set(COMPARISON_SPECS) - set(comparisons))}, "
            f"unexpected={sorted(set(comparisons) - set(COMPARISON_SPECS))}"
        )
    for name, specification in COMPARISON_SPECS.items():
        context: str = f"paired-comparisons.comparisons.{name}"
        comparison: JsonObject = require_object(comparisons.get(name), context)
        multi_model: bool = len(specification["models"]) > 1
        fields: frozenset[str] = frozenset(
            {
                "schema_version",
                "split",
                "arm_a",
                "arm_b",
                "models",
                "by_model",
                *(("fleet_hierarchical",) if multi_model else ()),
            }
        )
        require_exact_keys(comparison, fields, context)
        if comparison.get("schema_version") != (
            "k2-answer-paired-comparison-v1"
        ):
            raise DownstreamDataError(
                f"Unexpected paired comparison schema: context={context}"
            )
        if comparison.get("split") != "heldout":
            raise DownstreamDataError(
                f"Paired comparison split mismatch: context={context}"
            )
        if (
            comparison.get("arm_a") != specification["arm_a"]
            or comparison.get("arm_b") != specification["arm_b"]
        ):
            raise DownstreamDataError(
                f"Paired comparison arms mismatch: context={context}"
            )
        if require_string_list(
            comparison.get("models"),
            f"{context}.models",
        ) != specification["models"]:
            raise DownstreamDataError(
                f"Paired comparison support mismatch: context={context}"
            )
        by_model: JsonObject = require_object(
            comparison.get("by_model"),
            f"{context}.by_model",
        )
        if set(by_model) != set(specification["models"]):
            raise DownstreamDataError(
                f"Paired by-model support mismatch: context={context}"
            )
        for model in specification["models"]:
            model_context: str = f"{context}.by_model.{model}"
            model_block: JsonObject = require_object(
                by_model.get(model),
                model_context,
            )
            require_exact_keys(
                model_block,
                frozenset({"by_domain", "pooled_instances"}),
                model_context,
            )
            by_domain: JsonObject = require_object(
                model_block.get("by_domain"),
                f"{model_context}.by_domain",
            )
            if set(by_domain) != set(DOMAINS):
                raise DownstreamDataError(
                    f"Paired domain support mismatch: context={model_context}"
                )
            for domain in DOMAINS:
                validate_bootstrap_stats(
                    by_domain.get(domain),
                    DOMAIN_HELDOUT_COUNTS[domain],
                    answer_accuracy(
                        answer_index,
                        "model_domain",
                        "heldout",
                        model,
                        domain,
                        specification["arm_a"],
                    ),
                    answer_accuracy(
                        answer_index,
                        "model_domain",
                        "heldout",
                        model,
                        domain,
                        specification["arm_b"],
                    ),
                    f"{model_context}.by_domain.{domain}",
                )
            validate_bootstrap_stats(
                model_block.get("pooled_instances"),
                EXPECTED_HELDOUT_PER_MODEL,
                answer_accuracy(
                    answer_index,
                    "model_pooled",
                    "heldout",
                    model,
                    None,
                    specification["arm_a"],
                ),
                answer_accuracy(
                    answer_index,
                    "model_pooled",
                    "heldout",
                    model,
                    None,
                    specification["arm_b"],
                ),
                f"{model_context}.pooled_instances",
            )
        if multi_model:
            validate_hierarchical(
                comparison.get("fleet_hierarchical"),
                specification,
                by_model,
                f"{context}.fleet_hierarchical",
            )


def write_json(path: Path, payload: JsonLike) -> None:
    """Write one sorted, formatted JSON value."""

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def canonical_jsonl_bytes(rows: Sequence[JsonObject]) -> bytes:
    """Return the frozen canonical JSONL representation."""

    return "".join(
        canonical_json(row) + "\n" for row in rows
    ).encode("utf-8")


def require_canonical_jsonl_source(
    path: Path,
    rows: Sequence[JsonObject],
    context: str,
) -> bytes:
    """Require input bytes to match the validated canonical row sequence."""

    source_bytes: bytes = path.read_bytes()
    expected_bytes: bytes = canonical_jsonl_bytes(rows)
    if source_bytes != expected_bytes:
        raise DownstreamDataError(
            "Fleet JSONL is not in its canonical producer representation: "
            f"context={context}, path={path}"
        )
    return source_bytes


def write_gzip_bytes(path: Path, content: bytes) -> None:
    """Write deterministic level-6 gzip bytes with a frozen Unix header."""

    compressed: bytearray = bytearray(
        gzip.compress(content, compresslevel=6, mtime=0)
    )
    compressed[GZIP_OPERATING_SYSTEM_OFFSET] = GZIP_OPERATING_SYSTEM_UNIX
    path.write_bytes(bytes(compressed))


def output_record(
    directory: Path,
    filename: str,
    rows: int,
    schema: str,
) -> JsonObject:
    """Return one path-free generated-file manifest record."""

    path: Path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Generated fleet file is missing: path={path}"
        )
    return {
        "sha256": sha256_file(path),
        "rows": rows,
        "schema": schema,
    }


def public_summary(
    loading_fleet: Sequence[JsonObject],
    answer_model_pooled: JsonObject,
    answer_fleet: JsonObject,
) -> JsonObject:
    """Build the path-free public statistical summary."""

    return {
        "schema_version": PUBLIC_SUMMARY_SCHEMA,
        "k_samples": 2,
        "loading": {
            "schema_version": "k2-loading-summary-v1",
            "valid": True,
            "input_records": EXPECTED_LOADING_DECISIONS,
            "expected_total_per_model": EXPECTED_TOTAL_PER_MODEL,
            "expected_heldout_per_model": EXPECTED_HELDOUT_PER_MODEL,
            "seven_model_support": list(SEVEN_MODELS),
            "five_model_support": list(FIVE_MODELS),
            "fleet_model_macro": list(loading_fleet),
        },
        "answers": {
            "schema_version": "k2-answer-summary-v1",
            "model_pooled": answer_model_pooled,
            "fleet_model_macro": answer_fleet,
            "seven_models": list(SEVEN_MODELS),
            "five_models": list(FIVE_MODELS),
            "fixed_model": FIXED_MODEL,
            "expected_total_per_model": EXPECTED_TOTAL_PER_MODEL,
            "expected_heldout_per_model": EXPECTED_HELDOUT_PER_MODEL,
        },
        "comparison_scope": (
            "completed K=2 main-experiment contrasts only; legacy baseline "
            "contrasts are excluded pending runtime-identity closure"
        ),
    }


def build_pack(
    output_dir: Path,
    input_paths: Mapping[str, Path],
    loading_rows: Sequence[JsonObject],
    answer_rows: Sequence[JsonObject],
    summary: JsonObject,
    comparisons: JsonObject,
) -> None:
    """Atomically publish exactly five deterministic fleet files."""

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing output directory: "
            f"path={output_dir}"
        )
    parent: Path = output_dir.parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: path={parent}"
        )
    loading_bytes: bytes = require_canonical_jsonl_source(
        input_paths["loading_metrics_long"],
        loading_rows,
        "loading-metrics",
    )
    answer_bytes: bytes = require_canonical_jsonl_source(
        input_paths["answer_metrics_long"],
        answer_rows,
        "answer-metrics",
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.tmp.",
        dir=parent,
    ) as temporary_name:
        temporary_dir: Path = Path(temporary_name)
        write_gzip_bytes(
            temporary_dir / "loading_metrics_long.jsonl.gz",
            loading_bytes,
        )
        write_gzip_bytes(
            temporary_dir / "answer_metrics_long.jsonl.gz",
            answer_bytes,
        )
        write_json(temporary_dir / "summary.json", summary)
        paired_output: Path = temporary_dir / "paired_comparisons.json"
        paired_source: Path = input_paths["paired_comparisons_current4"]
        paired_output.write_bytes(paired_source.read_bytes())
        if load_json_object(
            paired_output,
            "generated-paired-comparisons",
        ) != comparisons:
            raise DownstreamDataError(
                "Generated paired-comparisons bytes changed the payload"
            )
        files: JsonObject = {
            "answer_metrics_long.jsonl.gz": output_record(
                temporary_dir,
                "answer_metrics_long.jsonl.gz",
                EXPECTED_METRIC_ROWS,
                "k2-answer-metrics-long-v1",
            ),
            "loading_metrics_long.jsonl.gz": output_record(
                temporary_dir,
                "loading_metrics_long.jsonl.gz",
                EXPECTED_METRIC_ROWS,
                "k2-loading-metrics-long-v1",
            ),
            "paired_comparisons.json": output_record(
                temporary_dir,
                "paired_comparisons.json",
                len(COMPARISON_SPECS),
                "k2-answer-paired-comparisons-v1",
            ),
            "summary.json": output_record(
                temporary_dir,
                "summary.json",
                1,
                PUBLIC_SUMMARY_SCHEMA,
            ),
        }
        manifest: JsonObject = {
            "schema_version": PUBLIC_MANIFEST_SCHEMA,
            "k_samples": 2,
            "files": {
                filename: files[filename] for filename in sorted(files)
            },
            "support": {
                "seven_models": list(SEVEN_MODELS),
                "select_eligible_models": list(FIVE_MODELS),
                "unavailable_select_models": list(
                    SELECT_UNAVAILABLE_MODELS
                ),
            },
            "baseline_comparisons": {
                "status": "excluded_pending_runtime_identity_gate",
                "reason": (
                    "Legacy baseline comparisons are not part of this upload "
                    "until runtime identity is proven or controlled baseline "
                    "reruns finish."
                ),
            },
            "manifest_self_policy": {
                "included_in_files": False,
                "reason": (
                    "manifest.json is excluded because a file cannot contain "
                    "its own stable SHA-256 digest."
                ),
            },
        }
        write_json(temporary_dir / "manifest.json", manifest)
        actual_names: set[str] = {
            entry.name for entry in temporary_dir.iterdir()
        }
        expected_names: set[str] = set(OUTPUT_FILENAMES)
        if actual_names != expected_names:
            raise DownstreamDataError(
                "Generated fleet file set mismatch: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        os.rename(temporary_dir, output_dir)


def main() -> None:
    """Validate the private aggregates and export the public fleet pack."""

    args = parse_args()
    output_dir: Path = cast(Path, args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing output directory: "
            f"path={output_dir}"
        )
    input_paths: dict[str, Path] = {
        "answer_metrics_long": cast(
            Path,
            args.answer_metrics_long,
        ).resolve(),
        "answer_summary": cast(Path, args.answer_summary).resolve(),
        "loading_metrics_long": cast(
            Path,
            args.loading_metrics_long,
        ).resolve(),
        "loading_summary": cast(Path, args.loading_summary).resolve(),
        "paired_comparisons_current4": cast(
            Path,
            args.paired_comparisons_current4,
        ).resolve(),
    }
    loading_rows: list[JsonObject] = load_jsonl(
        input_paths["loading_metrics_long"],
        "loading-metrics",
    )
    answer_rows: list[JsonObject] = load_jsonl(
        input_paths["answer_metrics_long"],
        "answer-metrics",
    )
    loading_index: dict[MetricKey, JsonObject] = validate_loading_metrics(
        loading_rows
    )
    answer_index: dict[MetricKey, JsonObject] = validate_answer_metrics(
        answer_rows
    )
    loading_fleet: list[JsonObject] = loading_fleet_rows(loading_index)
    answer_model_pooled, answer_fleet = answer_summary_blocks(answer_index)
    loading_summary: JsonObject = load_json_object(
        input_paths["loading_summary"],
        "loading-summary",
    )
    answer_summary: JsonObject = load_json_object(
        input_paths["answer_summary"],
        "answer-summary",
    )
    comparisons: JsonObject = load_json_object(
        input_paths["paired_comparisons_current4"],
        "paired-comparisons",
    )
    validate_loading_summary(
        loading_summary,
        input_paths["loading_metrics_long"],
        loading_fleet,
    )
    validate_answer_summary(
        answer_summary,
        answer_model_pooled,
        answer_fleet,
    )
    validate_comparisons(comparisons, answer_index)
    summary: JsonObject = public_summary(
        loading_fleet,
        answer_model_pooled,
        answer_fleet,
    )
    build_pack(
        output_dir,
        input_paths,
        loading_rows,
        answer_rows,
        summary,
        comparisons,
    )
    print(
        canonical_json(
            {
                "event": "k2_fleet_public_pack_exported",
                "output_dir": str(output_dir),
                "manifest_sha256": sha256_file(
                    output_dir / "manifest.json"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
