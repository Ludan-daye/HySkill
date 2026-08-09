#!/usr/bin/env python3
"""Audit one complete four-domain runtime-matched Bare model run."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

from hyskill.runtime_matched_execution import (
    canonical_json,
    sha256_file,
)


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

ANSWER_SCHEMA_VERSION: str = "runtime-matched-baseline-answer-v1"
USAGE_SCHEMA_VERSION: str = "runtime-matched-usage-event-v1"
ATTEMPT_SCHEMA_VERSION: str = "runtime-matched-answer-attempt-v1"
EVALUATION_SCHEMA_VERSION: str = "runtime-matched-baseline-evaluation-v1"
AUDIT_SCHEMA_VERSION: str = "runtime-matched-bare-completeness-v1"
DOMAIN_COUNTS: tuple[tuple[str, int], ...] = (
    ("theoremqa", 747),
    ("logicbench", 760),
    ("medcalcbench", 1100),
    ("champ", 223),
)
RESOLVED_CATEGORIES: frozenset[str] = frozenset(
    {"success", "method_failure"}
)


class BareAuditError(ValueError):
    """Raised when a completed Bare model run violates its contract."""


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-level Bare audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--instances-dir", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return one JSON object or raise with source context."""

    if not isinstance(value, dict):
        raise BareAuditError(
            f"Expected JSON object: context={context}, "
            f"value_type={type(value).__name__}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return one JSON list or raise with source context."""

    if not isinstance(value, list):
        raise BareAuditError(
            f"Expected JSON list: context={context}, "
            f"value_type={type(value).__name__}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return one non-empty string or raise with source context."""

    if not isinstance(value, str) or not value:
        raise BareAuditError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_integer(value: JsonValue | None, context: str) -> int:
    """Return one non-negative integer or raise with source context."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BareAuditError(
            f"Expected non-negative integer: context={context}, value={value!r}"
        )
    return value


def load_json(path: Path, context: str) -> JsonValue:
    """Load one strict UTF-8 JSON file."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    try:
        return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise BareAuditError(
            f"{context} JSON is malformed: path={path}, "
            f"line={error.lineno}, column={error.colno}, message={error.msg}"
        ) from error


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load one non-empty strict UTF-8 JSONL file."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise BareAuditError(
                    f"{context} JSONL contains a blank line: "
                    f"path={path}, line={line_number}"
                )
            try:
                value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise BareAuditError(
                    f"{context} JSONL is malformed: path={path}, "
                    f"line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            rows.append(
                require_object(value, f"{context}:{path}:{line_number}")
            )
    if not rows:
        raise BareAuditError(f"{context} JSONL is empty: path={path}")
    return rows


def index_rows(
    rows: Sequence[JsonObject],
    context: str,
) -> dict[str, JsonObject]:
    """Index records by one required unique instance ID."""

    output: dict[str, JsonObject] = {}
    for row_number, row in enumerate(rows, start=1):
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}:{row_number}.instance_id",
        )
        if instance_id in output:
            raise BareAuditError(
                f"{context} contains a duplicate instance ID: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def load_instance_ids(
    path: Path,
    domain: str,
    expected_count: int,
) -> set[str]:
    """Load the exact unique instance support for one domain."""

    rows: list[JsonValue] = require_list(load_json(path, "instances"), "instances")
    if len(rows) != expected_count:
        raise BareAuditError(
            "Instance denominator mismatch: "
            f"domain={domain}, expected={expected_count}, actual={len(rows)}"
        )
    instance_ids: set[str] = set()
    for row_number, value in enumerate(rows, start=1):
        row: JsonObject = require_object(value, f"instances:{row_number}")
        instance_id: str = require_string(
            row.get("instance_id"),
            f"instances:{row_number}.instance_id",
        )
        if instance_id in instance_ids:
            raise BareAuditError(
                f"Instances contain a duplicate ID: instance_id={instance_id}"
            )
        instance_domain: str = require_string(
            row.get("dataset"),
            f"instances:{row_number}.dataset",
        )
        if instance_domain != domain:
            raise BareAuditError(
                "Instance domain mismatch: "
                f"instance_id={instance_id}, expected={domain}, "
                f"actual={instance_domain}"
            )
        instance_ids.add(instance_id)
    return instance_ids


def validate_answers(
    path: Path,
    expected_ids: set[str],
    result_tag: str,
    served_model: str,
    domain: str,
) -> tuple[dict[str, JsonObject], dict[str, int]]:
    """Validate complete fresh-only terminal answer rows."""

    answers: dict[str, JsonObject] = index_rows(
        load_jsonl(path, "answers"),
        "answers",
    )
    if set(answers) != expected_ids:
        raise BareAuditError(
            "Answer coverage mismatch: "
            f"domain={domain}, missing={sorted(expected_ids - set(answers))[:20]}, "
            f"unexpected={sorted(set(answers) - expected_ids)[:20]}"
        )
    categories: dict[str, int] = {}
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("schema_version", ANSWER_SCHEMA_VERSION),
        ("stage", "answer"),
        ("model", result_tag),
        ("served_model", served_model),
        ("domain", domain),
        ("dataset", domain),
        ("arm", "bare"),
        ("reused_same_arm", False),
    )
    for instance_id, answer in answers.items():
        mismatches: list[str] = [
            f"{field_name}:expected={expected!r},"
            f"actual={answer.get(field_name)!r}"
            for field_name, expected in expected_fields
            if answer.get(field_name) != expected
        ]
        if mismatches:
            raise BareAuditError(
                "Answer identity mismatch: "
                f"instance_id={instance_id}, mismatches={mismatches}"
            )
        category: str = require_string(
            answer.get("failure_category"),
            f"answers:{instance_id}.failure_category",
        )
        if category not in RESOLVED_CATEGORIES:
            raise BareAuditError(
                "Answer has an unresolved failure category: "
                f"instance_id={instance_id}, category={category}"
            )
        categories[category] = categories.get(category, 0) + 1
    return answers, categories


def validate_usage(
    path: Path,
    expected_ids: set[str],
    served_model: str,
    domain: str,
) -> JsonObject:
    """Validate usage attribution and sum actual service-reported tokens."""

    rows: list[JsonObject] = load_jsonl(path, "usage")
    observed_ids: set[str] = set()
    event_ids: set[tuple[str, int, int]] = set()
    statuses: dict[str, int] = {}
    missing_reasons: dict[str, int] = {}
    token_totals: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for row_number, row in enumerate(rows, start=1):
        instance_id: str = require_string(
            row.get("instance_id"),
            f"usage:{row_number}.instance_id",
        )
        if instance_id not in expected_ids:
            raise BareAuditError(
                "Usage event is outside the instance support: "
                f"domain={domain}, instance_id={instance_id}"
            )
        expected_fields: tuple[tuple[str, JsonValue], ...] = (
            ("schema_version", USAGE_SCHEMA_VERSION),
            ("model", served_model),
            ("domain", domain),
            ("arm", "bare"),
        )
        mismatches: list[str] = [
            f"{field_name}:expected={expected!r},actual={row.get(field_name)!r}"
            for field_name, expected in expected_fields
            if row.get(field_name) != expected
        ]
        if mismatches:
            raise BareAuditError(
                "Usage identity mismatch: "
                f"instance_id={instance_id}, mismatches={mismatches}"
            )
        logical_attempt: int = require_integer(
            row.get("logical_attempt"),
            f"usage:{row_number}.logical_attempt",
        )
        http_subcall: int = require_integer(
            row.get("http_subcall"),
            f"usage:{row_number}.http_subcall",
        )
        if logical_attempt == 0 or http_subcall == 0:
            raise BareAuditError(
                "Usage attempt identifiers must be positive: "
                f"instance_id={instance_id}, logical_attempt={logical_attempt}, "
                f"http_subcall={http_subcall}"
            )
        event_id: tuple[str, int, int] = (
            instance_id,
            logical_attempt,
            http_subcall,
        )
        if event_id in event_ids:
            raise BareAuditError(
                f"Usage contains a duplicate event: event_id={event_id}"
            )
        event_ids.add(event_id)
        observed_ids.add(instance_id)
        status: str = require_string(
            row.get("status"),
            f"usage:{row_number}.status",
        )
        if status not in {"response", "error"}:
            raise BareAuditError(
                f"Usage status is invalid: event_id={event_id}, status={status}"
            )
        statuses[status] = statuses.get(status, 0) + 1
        raw_reason: JsonValue | None = row.get("usage_missing_reason")
        reason: str | None
        if raw_reason is None:
            reason = None
        else:
            reason = require_string(
                raw_reason,
                f"usage:{row_number}.usage_missing_reason",
            )
            missing_reasons[reason] = missing_reasons.get(reason, 0) + 1
        for field_name in token_totals:
            raw_tokens: JsonValue | None = row.get(field_name)
            if raw_tokens is None:
                if reason is None:
                    raise BareAuditError(
                        "Null token usage requires a reason: "
                        f"event_id={event_id}, field={field_name}"
                    )
                continue
            token_totals[field_name] += require_integer(
                raw_tokens,
                f"usage:{row_number}.{field_name}",
            )
    if observed_ids != expected_ids:
        raise BareAuditError(
            "Terminal answers lack bound usage events: "
            f"domain={domain}, missing={sorted(expected_ids - observed_ids)[:20]}"
        )
    return {
        "events": len(rows),
        "statuses": statuses,
        "usage_missing_reasons": missing_reasons,
        "actual_token_totals": token_totals,
        "all_terminal_answers_have_usage": True,
    }


def validate_attempts(
    path: Path,
    expected_ids: set[str],
    served_model: str,
    domain: str,
) -> JsonObject:
    """Validate logical attempt coverage and identity."""

    rows: list[JsonObject] = load_jsonl(path, "attempts")
    observed_ids: set[str] = set()
    statuses: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=1):
        instance_id: str = require_string(
            row.get("instance_id"),
            f"attempts:{row_number}.instance_id",
        )
        if instance_id not in expected_ids:
            raise BareAuditError(
                "Attempt is outside the instance support: "
                f"domain={domain}, instance_id={instance_id}"
            )
        expected_fields: tuple[tuple[str, JsonValue], ...] = (
            ("schema_version", ATTEMPT_SCHEMA_VERSION),
            ("model", served_model),
            ("domain", domain),
            ("arm", "bare"),
        )
        mismatches: list[str] = [
            f"{field_name}:expected={expected!r},actual={row.get(field_name)!r}"
            for field_name, expected in expected_fields
            if row.get(field_name) != expected
        ]
        if mismatches:
            raise BareAuditError(
                "Attempt identity mismatch: "
                f"instance_id={instance_id}, mismatches={mismatches}"
            )
        observed_ids.add(instance_id)
        status: str = require_string(
            row.get("status"),
            f"attempts:{row_number}.status",
        )
        statuses[status] = statuses.get(status, 0) + 1
    if observed_ids != expected_ids:
        raise BareAuditError(
            "Terminal answers lack bound logical attempts: "
            f"domain={domain}, missing={sorted(expected_ids - observed_ids)[:20]}"
        )
    return {"events": len(rows), "statuses": statuses}


def load_terminal_summary(path: Path, domain: str) -> JsonObject:
    """Load and validate the terminal full-run summary from one job log."""

    if not path.is_file():
        raise FileNotFoundError(f"Full-run log does not exist: path={path}")
    nonempty_lines: list[str] = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not nonempty_lines:
        raise BareAuditError(f"Full-run log is empty: path={path}")
    try:
        summary: JsonObject = require_object(
            cast(JsonValue, json.loads(nonempty_lines[-1])),
            f"full-run-summary:{domain}",
        )
    except json.JSONDecodeError as error:
        raise BareAuditError(
            f"Full-run terminal line is malformed: path={path}, "
            f"message={error.msg}"
        ) from error
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("event", "runtime_matched_bare_job_state"),
        ("domain", domain),
        ("arm", "bare"),
        ("run_mode", "full"),
        ("missing_after_run", 0),
        ("unresolved", 0),
        ("reused_same_arm", 0),
        ("run_valid", True),
    )
    mismatches: list[str] = [
        f"{field_name}:expected={expected!r},actual={summary.get(field_name)!r}"
        for field_name, expected in expected_fields
        if summary.get(field_name) != expected
    ]
    if mismatches:
        raise BareAuditError(
            f"Full-run summary is invalid: domain={domain}, mismatches={mismatches}"
        )
    return summary


def validate_evaluation(
    path: Path,
    expected_ids: set[str],
    result_tag: str,
    served_model: str,
    domain: str,
) -> JsonObject:
    """Validate one production evaluation product and exact detail support."""

    payload: JsonObject = require_object(
        load_json(path, "evaluation"),
        "evaluation",
    )
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("schema_version", EVALUATION_SCHEMA_VERSION),
        ("model", result_tag),
        ("served_model", served_model),
        ("domain", domain),
        ("arm", "bare"),
    )
    mismatches: list[str] = [
        f"{field_name}:expected={expected!r},actual={payload.get(field_name)!r}"
        for field_name, expected in expected_fields
        if payload.get(field_name) != expected
    ]
    if mismatches:
        raise BareAuditError(
            f"Evaluation identity mismatch: domain={domain}, mismatches={mismatches}"
        )
    details: list[JsonValue] = require_list(
        payload.get("details"),
        "evaluation.details",
    )
    detail_rows: list[JsonObject] = [
        require_object(value, f"evaluation.details[{index}]")
        for index, value in enumerate(details)
    ]
    detail_index: dict[str, JsonObject] = index_rows(
        detail_rows,
        "evaluation.details",
    )
    if set(detail_index) != expected_ids:
        raise BareAuditError(
            "Evaluation detail coverage mismatch: "
            f"domain={domain}, "
            f"missing={sorted(expected_ids - set(detail_index))[:20]}, "
            f"unexpected={sorted(set(detail_index) - expected_ids)[:20]}"
        )
    metrics: JsonObject = require_object(
        payload.get("metrics"),
        "evaluation.metrics",
    )
    full_metrics: JsonObject = require_object(
        metrics.get("full"),
        "evaluation.metrics.full",
    )
    if full_metrics.get("total") != len(expected_ids):
        raise BareAuditError(
            "Evaluation full denominator mismatch: "
            f"domain={domain}, expected={len(expected_ids)}, "
            f"actual={full_metrics.get('total')!r}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "metrics": metrics,
    }


def domain_audit(
    result_root: Path,
    instances_dir: Path,
    result_tag: str,
    served_model: str,
    domain: str,
    expected_count: int,
) -> JsonObject:
    """Audit one domain and return its complete evidence summary."""

    instances_path: Path = instances_dir / f"{domain}.json"
    answers_path: Path = result_root / "answers" / f"{domain}-bare.jsonl"
    usage_path: Path = result_root / "logs" / f"{domain}-bare.usage.jsonl"
    attempts_path: Path = result_root / "logs" / f"{domain}-bare.attempts.jsonl"
    full_log_path: Path = result_root / "logs" / f"{domain}-bare.full.log"
    evaluation_path: Path = result_root / "eval" / f"{domain}-bare.eval.json"
    expected_ids: set[str] = load_instance_ids(
        instances_path,
        domain,
        expected_count,
    )
    _answers, categories = validate_answers(
        answers_path,
        expected_ids,
        result_tag,
        served_model,
        domain,
    )
    usage: JsonObject = validate_usage(
        usage_path,
        expected_ids,
        served_model,
        domain,
    )
    attempts: JsonObject = validate_attempts(
        attempts_path,
        expected_ids,
        served_model,
        domain,
    )
    full_summary: JsonObject = load_terminal_summary(full_log_path, domain)
    if (
        full_summary.get("expected_total") != expected_count
        or full_summary.get("observed_total") != expected_count
    ):
        raise BareAuditError(
            "Full-run summary denominator mismatch: "
            f"domain={domain}, expected={expected_count}, "
            f"expected_total={full_summary.get('expected_total')!r}, "
            f"observed_total={full_summary.get('observed_total')!r}"
        )
    evaluation: JsonObject = validate_evaluation(
        evaluation_path,
        expected_ids,
        result_tag,
        served_model,
        domain,
    )
    return {
        "expected_rows": expected_count,
        "observed_rows": expected_count,
        "failure_categories": categories,
        "answers": {
            "path": str(answers_path),
            "sha256": sha256_file(answers_path),
        },
        "usage": usage,
        "attempts": attempts,
        "full_run_summary": full_summary,
        "evaluation": evaluation,
        "valid": True,
    }


def aggregate_counts(
    domains: Mapping[str, JsonObject],
    nested_field: str,
) -> dict[str, int]:
    """Sum one nested integer-count object over all domains."""

    totals: dict[str, int] = {}
    for domain, payload in domains.items():
        raw_counts: JsonObject = require_object(
            payload.get(nested_field),
            f"domains.{domain}.{nested_field}",
        )
        for key, value in raw_counts.items():
            totals[key] = totals.get(key, 0) + require_integer(
                value,
                f"domains.{domain}.{nested_field}.{key}",
            )
    return totals


def aggregate_token_totals(
    token_domains: Mapping[str, JsonObject],
) -> dict[str, int]:
    """Sum actual token counts over all four domains."""

    return {
        field_name: sum(
            require_integer(
                token_payload.get(field_name),
                f"domains.{domain}.usage.actual_token_totals.{field_name}",
            )
            for domain, token_payload in token_domains.items()
        )
        for field_name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }


def write_json_atomic(path: Path, payload: JsonObject) -> None:
    """Write one JSON object atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    """Audit all four Bare domains and emit one model-level product."""

    args = parse_args()
    result_root: Path = cast(Path, args.result_root).resolve()
    instances_dir: Path = cast(Path, args.instances_dir).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    result_tag: str = str(args.result_tag)
    served_model: str = str(args.served_model)
    domains: dict[str, JsonObject] = {
        domain: domain_audit(
            result_root,
            instances_dir,
            result_tag,
            served_model,
            domain,
            expected_count,
        )
        for domain, expected_count in DOMAIN_COUNTS
    }
    token_domains: dict[str, JsonObject] = {
        domain: require_object(
            require_object(payload.get("usage"), f"domains.{domain}.usage").get(
                "actual_token_totals"
            ),
            f"domains.{domain}.usage.actual_token_totals",
        )
        for domain, payload in domains.items()
    }
    payload: JsonObject = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "model": result_tag,
        "served_model": served_model,
        "arm": "bare",
        "expected_rows": sum(count for _domain, count in DOMAIN_COUNTS),
        "observed_rows": sum(
            require_integer(
                domain_payload.get("observed_rows"),
                f"domains.{domain}.observed_rows",
            )
            for domain, domain_payload in domains.items()
        ),
        "failure_categories": aggregate_counts(
            domains,
            "failure_categories",
        ),
        "actual_token_totals": aggregate_token_totals(token_domains),
        "domains": domains,
        "fresh_only": True,
        "reused_same_arm": 0,
        "unresolved": 0,
        "valid": True,
    }
    if payload["observed_rows"] != payload["expected_rows"]:
        raise BareAuditError(
            "Model-level Bare denominator mismatch: "
            f"expected={payload['expected_rows']}, "
            f"observed={payload['observed_rows']}"
        )
    write_json_atomic(output_path, payload)
    print(
        canonical_json(
            {
                "event": "runtime_matched_bare_audit_complete",
                "model": result_tag,
                "expected_rows": payload["expected_rows"],
                "observed_rows": payload["observed_rows"],
                "failure_categories": payload["failure_categories"],
                "actual_token_totals": payload["actual_token_totals"],
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "valid": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
