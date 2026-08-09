#!/usr/bin/env python3
"""Summarize actual per-response usage for runtime-matched baselines."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict, cast

from hyskill.runtime_matched_execution import canonical_json, sha256_file


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
StageName: TypeAlias = Literal["decision", "answer"]

SCHEMA_VERSION: str = "runtime-matched-usage-summary-v1"
USAGE_EVENT_SCHEMA_VERSION: str = "runtime-matched-usage-event-v1"
DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
)
NATIVE_ARMS: tuple[str, ...] = ("always_rerank", "select_bm25")
UNDERSCORE_SELECT_FILE_MODELS: frozenset[str] = frozenset(
    {"llama31-8b"}
)


class UsageSummaryError(ValueError):
    """Raised when usage evidence is absent, malformed, or inconsistent."""


class UsageLogSpec(TypedDict):
    """One expected usage log and its semantic identity."""

    model: str
    domain: str
    arm: str
    stage: StageName
    path: Path


class UsageEvent(TypedDict):
    """Validated fields used from one attributed HTTP usage event."""

    served_model: str
    job_id: str
    domain: str
    arm: str
    instance_id: str
    logical_attempt: int
    http_subcall: int
    status: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    usage_missing_reason: str | None
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse explicit result inventory and output arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--seven-models", required=True)
    parser.add_argument("--five-models", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_model_list(value: str, context: str) -> tuple[str, ...]:
    """Parse one non-empty, ordered, duplicate-free model list."""

    models: tuple[str, ...] = tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    if not models:
        raise ValueError(f"{context} must contain at least one model")
    if len(models) != len(set(models)):
        raise ValueError(
            f"{context} contains duplicate models: models={models}"
        )
    return models


def native_file_label(model: str, arm: str) -> str:
    """Return the explicitly frozen on-disk label for one native arm."""

    if arm == "always_rerank":
        return arm
    if arm != "select_bm25":
        raise ValueError(f"Unsupported native arm: arm={arm}")
    if model in UNDERSCORE_SELECT_FILE_MODELS:
        return "select_bm25"
    return "select-bm25"


def usage_log_specs(
    result_root: Path,
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> list[UsageLogSpec]:
    """Return the frozen 108-job usage inventory."""

    eligible_models: set[str] = set(five_models)
    specs: list[UsageLogSpec] = []
    for model in seven_models:
        logs_root: Path = result_root / model / "logs"
        for domain in DOMAINS:
            specs.append(
                {
                    "model": model,
                    "domain": domain,
                    "arm": "bare",
                    "stage": "answer",
                    "path": logs_root / f"{domain}-bare.usage.jsonl",
                }
            )
            if model not in eligible_models:
                continue
            for arm in NATIVE_ARMS:
                file_label: str = native_file_label(model, arm)
                specs.extend(
                    (
                        {
                            "model": model,
                            "domain": domain,
                            "arm": arm,
                            "stage": "decision",
                            "path": (
                                logs_root
                                / (
                                    f"{domain}-{file_label}."
                                    "decision.attempts.jsonl"
                                )
                            ),
                        },
                        {
                            "model": model,
                            "domain": domain,
                            "arm": arm,
                            "stage": "answer",
                            "path": (
                                logs_root
                                / (
                                    f"{domain}-{file_label}."
                                    "answer.attempts.jsonl"
                                )
                            ),
                        },
                    )
                )
    return specs


def require_object(value: object, context: str) -> JsonObject:
    """Return one JSON object with source context."""

    if not isinstance(value, dict):
        raise UsageSummaryError(
            "Expected JSON object: "
            f"context={context}, type={type(value).__name__}"
        )
    return cast(JsonObject, value)


def require_string(value: JsonValue | None, context: str) -> str:
    """Return one non-empty string."""

    if not isinstance(value, str) or not value:
        raise UsageSummaryError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_positive_integer(value: JsonValue | None, context: str) -> int:
    """Return one positive integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise UsageSummaryError(
            f"Expected positive integer: context={context}, value={value!r}"
        )
    return value


def optional_nonnegative_integer(
    value: JsonValue | None,
    context: str,
) -> int | None:
    """Return one optional nonnegative integer."""

    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise UsageSummaryError(
            "Expected null or nonnegative integer: "
            f"context={context}, value={value!r}"
        )
    return value


def require_nonnegative_number(
    value: JsonValue | None,
    context: str,
) -> float:
    """Return one nonnegative finite elapsed duration."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
    ):
        raise UsageSummaryError(
            f"Expected nonnegative number: context={context}, value={value!r}"
        )
    output: float = float(value)
    if output == float("inf") or output != output:
        raise UsageSummaryError(
            f"Elapsed duration is not finite: context={context}, value={value!r}"
        )
    return output


def validate_token_usage(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    missing_reason: str | None,
    context: str,
) -> None:
    """Validate reported tokens or an explicit all-null reason."""

    values: tuple[int | None, int | None, int | None] = (
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )
    if all(value is None for value in values):
        if missing_reason is None or not missing_reason:
            raise UsageSummaryError(
                "All-null usage requires a non-empty reason: "
                f"context={context}"
            )
        return
    if any(value is None for value in values):
        raise UsageSummaryError(
            f"Usage fields are partially missing: context={context}"
        )
    if missing_reason is not None:
        raise UsageSummaryError(
            "Complete token usage cannot carry a missing reason: "
            f"context={context}, reason={missing_reason!r}"
        )
    if cast(int, prompt_tokens) + cast(int, completion_tokens) != total_tokens:
        raise UsageSummaryError(
            "Token usage total is inconsistent: "
            f"context={context}, prompt={prompt_tokens}, "
            f"completion={completion_tokens}, total={total_tokens}"
        )


def parse_usage_event(
    row: Mapping[str, JsonValue],
    spec: UsageLogSpec,
    context: str,
) -> UsageEvent:
    """Validate one usage event against its expected job identity."""

    domain: str = require_string(row.get("domain"), f"{context}.domain")
    arm: str = require_string(row.get("arm"), f"{context}.arm")
    if domain != spec["domain"] or arm != spec["arm"]:
        raise UsageSummaryError(
            "Usage event identity differs from its log: "
            f"context={context}, expected_domain={spec['domain']}, "
            f"actual_domain={domain}, expected_arm={spec['arm']}, "
            f"actual_arm={arm}"
        )
    status: str = require_string(row.get("status"), f"{context}.status")
    if status not in ("response", "error"):
        raise UsageSummaryError(
            f"Unexpected usage status: context={context}, status={status!r}"
        )
    prompt_tokens: int | None = optional_nonnegative_integer(
        row.get("prompt_tokens"),
        f"{context}.prompt_tokens",
    )
    completion_tokens: int | None = optional_nonnegative_integer(
        row.get("completion_tokens"),
        f"{context}.completion_tokens",
    )
    total_tokens: int | None = optional_nonnegative_integer(
        row.get("total_tokens"),
        f"{context}.total_tokens",
    )
    raw_reason: JsonValue | None = row.get("usage_missing_reason")
    if raw_reason is not None and not isinstance(raw_reason, str):
        raise UsageSummaryError(
            "Usage missing reason must be null or a string: "
            f"context={context}, value={raw_reason!r}"
        )
    missing_reason: str | None = cast(str | None, raw_reason)
    validate_token_usage(
        prompt_tokens,
        completion_tokens,
        total_tokens,
        missing_reason,
        context,
    )
    return {
        "served_model": require_string(
            row.get("model"),
            f"{context}.model",
        ),
        "job_id": require_string(row.get("job_id"), f"{context}.job_id"),
        "domain": domain,
        "arm": arm,
        "instance_id": require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        ),
        "logical_attempt": require_positive_integer(
            row.get("logical_attempt"),
            f"{context}.logical_attempt",
        ),
        "http_subcall": require_positive_integer(
            row.get("http_subcall"),
            f"{context}.http_subcall",
        ),
        "status": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "usage_missing_reason": missing_reason,
        "elapsed_seconds": require_nonnegative_number(
            row.get("elapsed_seconds"),
            f"{context}.elapsed_seconds",
        ),
    }


def load_usage_events(spec: UsageLogSpec) -> list[UsageEvent]:
    """Load all usage events from one mixed attempt log."""

    path: Path = spec["path"]
    if not path.is_file():
        raise FileNotFoundError(
            "Expected runtime-matched usage log does not exist: "
            f"model={spec['model']}, domain={spec['domain']}, "
            f"arm={spec['arm']}, stage={spec['stage']}, path={path}"
        )
    events: list[UsageEvent] = []
    seen_keys: set[tuple[str, str, int, int]] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_value: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise UsageSummaryError(
                    "Usage log contains malformed JSON: "
                    f"path={path}, line={line_number}, "
                    f"column={error.colno}, message={error.msg}"
                ) from error
            row: JsonObject = require_object(
                raw_value,
                f"{path}:{line_number}",
            )
            if row.get("schema_version") != USAGE_EVENT_SCHEMA_VERSION:
                continue
            event: UsageEvent = parse_usage_event(
                row,
                spec,
                f"{path}:{line_number}",
            )
            key: tuple[str, str, int, int] = (
                event["job_id"],
                event["instance_id"],
                event["logical_attempt"],
                event["http_subcall"],
            )
            if key in seen_keys:
                raise UsageSummaryError(
                    "Usage log contains a duplicate attributed HTTP call: "
                    f"path={path}, key={key}"
                )
            seen_keys.add(key)
            events.append(event)
    if not events:
        raise UsageSummaryError(
            "Expected usage log contains no usage events: "
            f"model={spec['model']}, domain={spec['domain']}, "
            f"arm={spec['arm']}, stage={spec['stage']}, path={path}"
        )
    return events


def summarize_events(events: Sequence[UsageEvent]) -> JsonObject:
    """Aggregate actual calls, reported tokens, and latency."""

    if not events:
        raise ValueError("Cannot summarize an empty usage event sequence")
    reported: list[UsageEvent] = [
        event for event in events if event["total_tokens"] is not None
    ]
    missing_reasons: dict[str, int] = {}
    for event in events:
        reason: str | None = event["usage_missing_reason"]
        if reason is None:
            continue
        missing_reasons[reason] = missing_reasons.get(reason, 0) + 1
    return {
        "http_calls": len(events),
        "response_calls": sum(
            event["status"] == "response" for event in events
        ),
        "error_calls": sum(event["status"] == "error" for event in events),
        "unique_instances": len(
            {event["instance_id"] for event in events}
        ),
        "usage_reported_calls": len(reported),
        "usage_missing_calls": len(events) - len(reported),
        "usage_missing_reasons": missing_reasons,
        "prompt_tokens": sum(
            cast(int, event["prompt_tokens"]) for event in reported
        ),
        "completion_tokens": sum(
            cast(int, event["completion_tokens"]) for event in reported
        ),
        "total_tokens": sum(
            cast(int, event["total_tokens"]) for event in reported
        ),
        "elapsed_seconds_sum": sum(
            event["elapsed_seconds"] for event in events
        ),
        "elapsed_seconds_median": statistics.median(
            event["elapsed_seconds"] for event in events
        ),
        "elapsed_seconds_max": max(
            event["elapsed_seconds"] for event in events
        ),
    }


def assert_globally_unique_calls(
    records: Sequence[tuple[UsageLogSpec, Sequence[UsageEvent]]],
) -> None:
    """Reject one attributed HTTP call appearing in multiple job logs."""

    seen: dict[tuple[str, str, int, int], Path] = {}
    for spec, events in records:
        for event in events:
            key: tuple[str, str, int, int] = (
                event["job_id"],
                event["instance_id"],
                event["logical_attempt"],
                event["http_subcall"],
            )
            previous_path: Path | None = seen.get(key)
            if previous_path is not None:
                raise UsageSummaryError(
                    "Attributed HTTP call appears in multiple usage logs: "
                    f"key={key}, first={previous_path}, second={spec['path']}"
                )
            seen[key] = spec["path"]


def aggregate_records(
    records: Sequence[tuple[UsageLogSpec, Sequence[UsageEvent]]],
    key_fields: Sequence[str],
) -> list[JsonObject]:
    """Aggregate job events under explicit identity fields."""

    groups: dict[tuple[str, ...], list[UsageEvent]] = {}
    for spec, events in records:
        raw_key: list[str] = []
        for field in key_fields:
            value: object = spec.get(field)
            if not isinstance(value, str):
                raise ValueError(
                    "Aggregate key field is not a string: "
                    f"field={field}, value={value!r}"
                )
            raw_key.append(value)
        groups.setdefault(tuple(raw_key), []).extend(events)
    output: list[JsonObject] = []
    for key in sorted(groups):
        identity: JsonObject = {
            field: value for field, value in zip(key_fields, key)
        }
        output.append({**identity, **summarize_events(groups[key])})
    return output


def write_json_atomic(path: Path, payload: JsonObject) -> None:
    """Write one JSON result atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    """Validate the frozen log inventory and write actual usage totals."""

    args = parse_args()
    result_root: Path = cast(Path, args.result_root).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    seven_models: tuple[str, ...] = parse_model_list(
        str(args.seven_models),
        "seven-models",
    )
    five_models: tuple[str, ...] = parse_model_list(
        str(args.five_models),
        "five-models",
    )
    if not set(five_models).issubset(seven_models):
        raise ValueError(
            "five-models must be a subset of seven-models: "
            f"seven={seven_models}, five={five_models}"
        )
    specs: list[UsageLogSpec] = usage_log_specs(
        result_root,
        seven_models,
        five_models,
    )
    expected_job_count: int = (
        len(seven_models) * len(DOMAINS)
        + len(five_models) * len(DOMAINS) * len(NATIVE_ARMS) * 2
    )
    if len(specs) != expected_job_count:
        raise AssertionError(
            "Internal usage inventory count mismatch: "
            f"expected={expected_job_count}, actual={len(specs)}"
        )
    records: list[tuple[UsageLogSpec, Sequence[UsageEvent]]] = [
        (spec, load_usage_events(spec)) for spec in specs
    ]
    assert_globally_unique_calls(records)
    source_logs: list[JsonObject] = [
        {
            "model": spec["model"],
            "domain": spec["domain"],
            "arm": spec["arm"],
            "stage": spec["stage"],
            "path": str(spec["path"]),
            "sha256": sha256_file(spec["path"]),
            **summarize_events(events),
        }
        for spec, events in records
    ]
    all_events: list[UsageEvent] = [
        event for _spec, events in records for event in events
    ]
    payload: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "support": {
            "seven_models": list(seven_models),
            "five_models": list(five_models),
            "domains": list(DOMAINS),
            "expected_job_logs": expected_job_count,
            "observed_job_logs": len(source_logs),
        },
        "overall": summarize_events(all_events),
        "by_model_arm_stage": aggregate_records(
            records,
            ("model", "arm", "stage"),
        ),
        "by_arm_stage": aggregate_records(
            records,
            ("arm", "stage"),
        ),
        "jobs": source_logs,
        "semantics": {
            "token_source": "actual OpenAI-compatible response usage",
            "http_calls_include_retries_and_tool_loop_subcalls": True,
            "missing_usage_is_never_imputed": True,
        },
    }
    write_json_atomic(output_path, payload)
    print(
        canonical_json(
            {
                "event": "runtime_matched_usage_summary_complete",
                "job_logs": len(source_logs),
                "http_calls": payload["overall"]["http_calls"],
                "total_tokens": payload["overall"]["total_tokens"],
                "usage_missing_calls": payload["overall"][
                    "usage_missing_calls"
                ],
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            }
        )
    )


if __name__ == "__main__":
    main()
