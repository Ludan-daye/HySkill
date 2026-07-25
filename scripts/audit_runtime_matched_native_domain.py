#!/usr/bin/env python3
"""Audit one completed native Rerank or BM25 Select domain."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, cast

from hyskill.runtime_matched_execution import (
    JobBoundManifest,
    canonical_json,
    load_job_bound_manifest,
    manifest_artifact,
    sha256_file,
    validate_frozen_k2_runtime_reference,
    verify_job_bound_manifest_files,
)
from hyskill.runtime_matched_rerank import (
    RERANK_DECISION_SCHEMA_VERSION,
)
from hyskill.runtime_matched_select import (
    SELECTION_RECORD_SCHEMA_VERSION,
)
from scripts.evaluate_runtime_matched_baselines import (
    ANSWER_SCHEMA_VERSION,
    EVALUATION_ROW_SCHEMA_VERSION,
    EVALUATION_SCHEMA_VERSION,
)
from scripts.summarize_runtime_matched_usage import (
    JsonObject,
    JsonValue,
    UsageEvent,
    UsageLogSpec,
    load_usage_events,
    summarize_events,
)


NativeArm: TypeAlias = Literal["always_rerank", "select_bm25"]

AUDIT_SCHEMA_VERSION: str = "runtime-matched-native-domain-audit-v1"
RESOLVED_ANSWER_CATEGORIES: frozenset[str] = frozenset(
    {"success", "method_failure"}
)
RESOLVED_DECISION_CATEGORIES: frozenset[str] = frozenset(
    {"success", "selector_fallback", "method_failure"}
)
NATIVE_ARMS: tuple[NativeArm, ...] = (
    "always_rerank",
    "select_bm25",
)


class NativeDomainAuditError(ValueError):
    """Raised when one native baseline domain lacks complete evidence."""


def parse_args() -> argparse.Namespace:
    """Parse one explicit native-domain audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--decision-attempt-log", required=True, type=Path)
    parser.add_argument("--answer-attempt-log", required=True, type=Path)
    parser.add_argument("--decision-full-log", required=True, type=Path)
    parser.add_argument("--answer-full-log", required=True, type=Path)
    parser.add_argument("--decision-manifest", required=True, type=Path)
    parser.add_argument("--answer-manifest", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--arm", required=True, choices=NATIVE_ARMS)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return one JSON object with source context."""

    if not isinstance(value, dict):
        raise NativeDomainAuditError(
            "Expected JSON object: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return one JSON list with source context."""

    if not isinstance(value, list):
        raise NativeDomainAuditError(
            "Expected JSON list: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return one non-empty string."""

    if not isinstance(value, str) or not value:
        raise NativeDomainAuditError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_boolean(value: JsonValue | None, context: str) -> bool:
    """Return one Boolean."""

    if not isinstance(value, bool):
        raise NativeDomainAuditError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


def require_nonnegative_integer(
    value: JsonValue | None,
    context: str,
) -> int:
    """Return one nonnegative integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise NativeDomainAuditError(
            f"Expected nonnegative integer: context={context}, value={value!r}"
        )
    return value


def load_json(path: Path, context: str) -> JsonObject:
    """Load one required UTF-8 JSON object."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: path={path}")
    try:
        value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise NativeDomainAuditError(
            f"{context} is malformed: path={path}, line={error.lineno}, "
            f"column={error.colno}, message={error.msg}"
        ) from error
    return require_object(value, context)


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load one required, non-empty UTF-8 JSONL file."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: path={path}")
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise NativeDomainAuditError(
                    f"{context} is malformed: path={path}, "
                    f"line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            rows.append(
                require_object(value, f"{context}:{line_number}")
            )
    if not rows:
        raise NativeDomainAuditError(
            f"{context} contains no records: path={path}"
        )
    return rows


def index_rows(
    rows: Sequence[JsonObject],
    context: str,
) -> dict[str, JsonObject]:
    """Index rows by unique instance ID."""

    output: dict[str, JsonObject] = {}
    for row_number, row in enumerate(rows, start=1):
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}:{row_number}.instance_id",
        )
        if instance_id in output:
            raise NativeDomainAuditError(
                f"{context} contains duplicate instance: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def load_instances(
    path: Path,
    domain: str,
    expected_count: int,
) -> dict[str, JsonObject]:
    """Load exact instance support with non-empty gold skill annotations."""

    if expected_count <= 0:
        raise ValueError(
            f"expected-count must be positive: value={expected_count}"
        )
    try:
        value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Instances do not exist: path={path}"
        ) from None
    except json.JSONDecodeError as error:
        raise NativeDomainAuditError(
            f"Instances are malformed: path={path}, line={error.lineno}, "
            f"column={error.colno}, message={error.msg}"
        ) from error
    rows: list[JsonObject] = [
        require_object(item, f"instances[{index}]")
        for index, item in enumerate(require_list(value, "instances"))
    ]
    if len(rows) != expected_count:
        raise NativeDomainAuditError(
            "Instance count mismatch: "
            f"domain={domain}, expected={expected_count}, actual={len(rows)}"
        )
    index: dict[str, JsonObject] = index_rows(rows, "instances")
    for instance_id, row in index.items():
        if row.get("dataset") != domain:
            raise NativeDomainAuditError(
                "Instance domain mismatch: "
                f"instance_id={instance_id}, expected={domain}, "
                f"actual={row.get('dataset')!r}"
            )
        gold_values: list[JsonValue] = require_list(
            row.get("skill_annotations"),
            f"instances:{instance_id}.skill_annotations",
        )
        gold_ids: list[str] = [
            require_string(
                item,
                f"instances:{instance_id}.skill_annotations[{gold_index}]",
            )
            for gold_index, item in enumerate(gold_values)
        ]
        if not gold_ids or len(gold_ids) != len(set(gold_ids)):
            raise NativeDomainAuditError(
                "Gold skill annotations must be non-empty and unique: "
                f"instance_id={instance_id}, values={gold_ids}"
            )
    return index


def validate_artifact_binding(
    manifest: JobBoundManifest,
    artifact_name: str,
    expected_path: Path,
) -> None:
    """Validate one manifest artifact against an exact local path."""

    evidence = manifest_artifact(manifest, artifact_name)
    resolved_path: Path = expected_path.resolve()
    if Path(evidence["path"]).resolve() != resolved_path:
        raise NativeDomainAuditError(
            "Manifest artifact path mismatch: "
            f"name={artifact_name}, expected={resolved_path}, "
            f"actual={evidence['path']}"
        )
    actual_size: int = resolved_path.stat().st_size
    actual_sha: str = sha256_file(resolved_path)
    if (
        evidence["size_bytes"] != actual_size
        or evidence["sha256"] != actual_sha
    ):
        raise NativeDomainAuditError(
            "Manifest artifact identity mismatch: "
            f"name={artifact_name}, expected_size={evidence['size_bytes']}, "
            f"actual_size={actual_size}, expected_sha={evidence['sha256']}, "
            f"actual_sha={actual_sha}"
        )


def validate_manifest(
    path: Path,
    repository_root: Path,
    instances_path: Path,
    decisions_path: Path | None,
    result_tag: str,
    served_model: str,
    domain: str,
    arm: NativeArm,
    stage: Literal["decision", "answer"],
) -> tuple[JobBoundManifest, str, str]:
    """Validate one job manifest and return it with SHA and job ID."""

    manifest: JobBoundManifest = load_job_bound_manifest(path)
    verify_job_bound_manifest_files(manifest, repository_root)
    validate_frozen_k2_runtime_reference(manifest["runtime_facts"])
    job: JsonObject = require_object(
        manifest["runtime_facts"].get("job"),
        "manifest.runtime_facts.job",
    )
    for field_name, expected in (
        ("result_tag", result_tag),
        ("model", served_model),
        ("domain", domain),
        ("arm", arm),
        ("stage", stage),
    ):
        if job.get(field_name) != expected:
            raise NativeDomainAuditError(
                "Manifest job identity mismatch: "
                f"field={field_name}, expected={expected!r}, "
                f"actual={job.get(field_name)!r}"
            )
    validate_artifact_binding(manifest, "instances", instances_path)
    if stage == "answer":
        if decisions_path is None:
            raise ValueError("Answer manifest validation requires decisions")
        decision_artifact: str = (
            "select_decisions"
            if arm == "select_bm25"
            else "rerank_decisions"
        )
        validate_artifact_binding(
            manifest,
            decision_artifact,
            decisions_path,
        )
    return (
        manifest,
        sha256_file(path),
        require_string(job.get("job_id"), "manifest.job.job_id"),
    )


def resolved_answer_category(
    row: Mapping[str, JsonValue],
    context: str,
) -> str:
    """Return one resolved answer category."""

    category: str = require_string(
        row.get("failure_category"),
        f"{context}.failure_category",
    )
    if category not in RESOLVED_ANSWER_CATEGORIES:
        raise NativeDomainAuditError(
            "Answer has an unresolved failure category: "
            f"context={context}, category={category}"
        )
    return category


def resolved_decision_category(
    row: Mapping[str, JsonValue],
    context: str,
) -> str:
    """Return one resolved native-decision category."""

    category: str = require_string(
        row.get("failure_category"),
        f"{context}.failure_category",
    )
    if category not in RESOLVED_DECISION_CATEGORIES:
        raise NativeDomainAuditError(
            "Decision has an unresolved failure category: "
            f"context={context}, category={category}"
        )
    return category


def count_categories(
    rows: Mapping[str, JsonObject],
) -> dict[str, int]:
    """Count already validated failure categories."""

    counts: dict[str, int] = {}
    for instance_id, row in rows.items():
        category: str = require_string(
            row.get("failure_category"),
            f"rows:{instance_id}.failure_category",
        )
        counts[category] = counts.get(category, 0) + 1
    return counts


def validate_decisions(
    rows: Sequence[JsonObject],
    expected_ids: set[str],
    result_tag: str,
    served_model: str,
    domain: str,
    arm: NativeArm,
    manifest: JobBoundManifest,
    manifest_sha: str,
) -> tuple[dict[str, JsonObject], dict[str, int]]:
    """Validate exact decision coverage and immutable runtime bindings."""

    index: dict[str, JsonObject] = index_rows(rows, "decisions")
    if set(index) != expected_ids:
        raise NativeDomainAuditError(
            "Decision coverage mismatch: "
            f"missing={sorted(expected_ids - set(index))[:20]}, "
            f"unexpected={sorted(set(index) - expected_ids)[:20]}"
        )
    expected_schema: str = (
        SELECTION_RECORD_SCHEMA_VERSION
        if arm == "select_bm25"
        else RERANK_DECISION_SCHEMA_VERSION
    )
    source_field: str = (
        "candidate_source_sha256"
        if arm == "select_bm25"
        else "source_sha256"
    )
    bm25_sha: str = manifest_artifact(
        manifest,
        "bm25_candidates",
    )["sha256"]
    code_bundle_sha: str = manifest["code_bundle_sha256"]
    call_counts: dict[str, int] = {}
    for instance_id, row in index.items():
        for field_name, expected in (
            ("schema_version", expected_schema),
            ("model", result_tag),
            ("served_model", served_model),
            ("domain", domain),
            ("arm", arm),
            ("stage", "decision"),
            ("runtime_manifest_sha256", manifest_sha),
            ("code_bundle_sha256", code_bundle_sha),
            (source_field, bm25_sha),
        ):
            if row.get(field_name) != expected:
                raise NativeDomainAuditError(
                    "Decision identity mismatch: "
                    f"instance_id={instance_id}, field={field_name}, "
                    f"expected={expected!r}, actual={row.get(field_name)!r}"
                )
        resolved_decision_category(row, f"decisions:{instance_id}")
        calls: int = require_nonnegative_integer(
            row.get("client_call_attempts"),
            f"decisions:{instance_id}.client_call_attempts",
        )
        if calls == 0:
            raise NativeDomainAuditError(
                "Native decision made no model call: "
                f"instance_id={instance_id}"
            )
        call_counts[instance_id] = calls
    return index, call_counts


def validate_answers(
    rows: Sequence[JsonObject],
    expected_ids: set[str],
    result_tag: str,
    served_model: str,
    domain: str,
    arm: NativeArm,
    decision_sha: str,
    manifest: JobBoundManifest,
    manifest_sha: str,
) -> tuple[dict[str, JsonObject], dict[str, int]]:
    """Validate exact answer coverage and return minimum HTTP-call counts."""

    index: dict[str, JsonObject] = index_rows(rows, "answers")
    if set(index) != expected_ids:
        raise NativeDomainAuditError(
            "Answer coverage mismatch: "
            f"missing={sorted(expected_ids - set(index))[:20]}, "
            f"unexpected={sorted(set(index) - expected_ids)[:20]}"
        )
    code_bundle_sha: str = manifest["code_bundle_sha256"]
    minimum_calls: dict[str, int] = {}
    for instance_id, row in index.items():
        for field_name, expected in (
            ("schema_version", ANSWER_SCHEMA_VERSION),
            ("model", result_tag),
            ("served_model", served_model),
            ("domain", domain),
            ("arm", arm),
            ("stage", "answer"),
            ("runtime_manifest_sha256", manifest_sha),
            ("code_bundle_sha256", code_bundle_sha),
            ("decision_source_sha256", decision_sha),
            ("reused_same_arm", False),
        ):
            if row.get(field_name) != expected:
                raise NativeDomainAuditError(
                    "Answer identity mismatch: "
                    f"instance_id={instance_id}, field={field_name}, "
                    f"expected={expected!r}, actual={row.get(field_name)!r}"
                )
        resolved_answer_category(row, f"answers:{instance_id}")
        if arm == "select_bm25":
            calls: int = require_nonnegative_integer(
                row.get("answer_call_attempts"),
                f"answers:{instance_id}.answer_call_attempts",
            )
        else:
            zero_call: bool = require_boolean(
                row.get("zero_call"),
                f"answers:{instance_id}.zero_call",
            )
            calls = require_nonnegative_integer(
                row.get("engine_attempts"),
                f"answers:{instance_id}.engine_attempts",
            )
            if zero_call != (calls == 0):
                raise NativeDomainAuditError(
                    "Rerank zero-call evidence is inconsistent: "
                    f"instance_id={instance_id}, zero_call={zero_call}, "
                    f"engine_attempts={calls}"
                )
        minimum_calls[instance_id] = calls
    return index, minimum_calls


def gold_ids(instance: JsonObject, instance_id: str) -> set[str]:
    """Return one instance's gold skill set."""

    return {
        require_string(
            value,
            f"instances:{instance_id}.skill_annotations[{index}]",
        )
        for index, value in enumerate(
            require_list(
                instance.get("skill_annotations"),
                f"instances:{instance_id}.skill_annotations",
            )
        )
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    """Return a finite ratio for a non-empty evaluation split."""

    if denominator <= 0:
        raise ValueError(
            f"Metric denominator must be positive: value={denominator}"
        )
    return numerator / denominator


def selection_metrics(
    instances: Mapping[str, JsonObject],
    decisions: Mapping[str, JsonObject],
    included_ids: set[str],
) -> JsonObject:
    """Compute candidate recall and the three frozen loading quantities."""

    if not included_ids:
        raise NativeDomainAuditError("Selection metric split is empty")
    loaded: int = 0
    gold_loaded: int = 0
    candidate_recalled: int = 0
    bm25_rank1_gold: int = 0
    for instance_id in sorted(included_ids):
        instance: JsonObject = instances[instance_id]
        decision: JsonObject = decisions[instance_id]
        gold: set[str] = gold_ids(instance, instance_id)
        raw_candidates: list[JsonValue] = require_list(
            decision.get("ordered_candidate_ids"),
            f"decisions:{instance_id}.ordered_candidate_ids",
        )
        candidates: list[str] = [
            require_string(
                item,
                f"decisions:{instance_id}.ordered_candidate_ids[{rank}]",
            )
            for rank, item in enumerate(raw_candidates)
        ]
        if len(candidates) != 50 or len(candidates) != len(set(candidates)):
            raise NativeDomainAuditError(
                "Native decision must bind 50 unique candidates: "
                f"instance_id={instance_id}, count={len(candidates)}"
            )
        if gold.intersection(candidates):
            candidate_recalled += 1
        if candidates[0] in gold:
            bm25_rank1_gold += 1
        category: str = resolved_decision_category(
            decision,
            f"decisions:{instance_id}",
        )
        selected_value: JsonValue | None = decision.get("selected_skill_id")
        if category in ("success", "selector_fallback"):
            selected: str = require_string(
                selected_value,
                f"decisions:{instance_id}.selected_skill_id",
            )
            loaded += 1
            if selected in gold:
                gold_loaded += 1
        elif selected_value is not None:
            raise NativeDomainAuditError(
                "Method-failure decision unexpectedly selected a skill: "
                f"instance_id={instance_id}, selected={selected_value!r}"
            )
    denominator: int = len(included_ids)
    return {
        "total": denominator,
        "loaded": loaded,
        "gold_loaded": gold_loaded,
        "candidate_recalled_at_50": candidate_recalled,
        "bm25_rank1_gold": bm25_rank1_gold,
        "loaded_skill_precision": (
            safe_ratio(gold_loaded, loaded) if loaded else 0.0
        ),
        "loading_rate": safe_ratio(loaded, denominator),
        "gold_load_rate": safe_ratio(gold_loaded, denominator),
        "candidate_recall_at_50": safe_ratio(
            candidate_recalled,
            denominator,
        ),
        "bm25_rank1_gold_rate": safe_ratio(
            bm25_rank1_gold,
            denominator,
        ),
    }


def validate_evaluation(
    path: Path,
    answers_path: Path,
    expected_ids: set[str],
    answers: Mapping[str, JsonObject],
    result_tag: str,
    served_model: str,
    domain: str,
    arm: NativeArm,
) -> tuple[JsonObject, set[str]]:
    """Validate evaluation coverage and return held-out IDs."""

    payload: JsonObject = load_json(path, "evaluation")
    for field_name, expected in (
        ("schema_version", EVALUATION_SCHEMA_VERSION),
        ("model", result_tag),
        ("served_model", served_model),
        ("domain", domain),
        ("arm", arm),
    ):
        if payload.get(field_name) != expected:
            raise NativeDomainAuditError(
                "Evaluation identity mismatch: "
                f"field={field_name}, expected={expected!r}, "
                f"actual={payload.get(field_name)!r}"
            )
    detail_rows: list[JsonObject] = [
        require_object(item, f"evaluation.details[{index}]")
        for index, item in enumerate(
            require_list(payload.get("details"), "evaluation.details")
        )
    ]
    details: dict[str, JsonObject] = index_rows(
        detail_rows,
        "evaluation.details",
    )
    if set(details) != expected_ids:
        raise NativeDomainAuditError(
            "Evaluation coverage mismatch: "
            f"missing={sorted(expected_ids - set(details))[:20]}, "
            f"unexpected={sorted(set(details) - expected_ids)[:20]}"
        )
    heldout_ids: set[str] = set()
    for instance_id, detail in details.items():
        for field_name, expected in (
            ("schema_version", EVALUATION_ROW_SCHEMA_VERSION),
            ("model", result_tag),
            ("served_model", served_model),
            ("domain", domain),
            ("arm", arm),
            (
                "failure_category",
                answers[instance_id]["failure_category"],
            ),
        ):
            if detail.get(field_name) != expected:
                raise NativeDomainAuditError(
                    "Evaluation row identity mismatch: "
                    f"instance_id={instance_id}, field={field_name}, "
                    f"expected={expected!r}, actual={detail.get(field_name)!r}"
                )
        if not require_boolean(
            detail.get("is_validation"),
            f"evaluation.details:{instance_id}.is_validation",
        ):
            heldout_ids.add(instance_id)
    metrics: JsonObject = require_object(
        payload.get("metrics"),
        "evaluation.metrics",
    )
    full_metrics: JsonObject = require_object(
        metrics.get("full"),
        "evaluation.metrics.full",
    )
    heldout_metrics: JsonObject = require_object(
        metrics.get("heldout"),
        "evaluation.metrics.heldout",
    )
    if (
        full_metrics.get("total") != len(expected_ids)
        or heldout_metrics.get("total") != len(heldout_ids)
    ):
        raise NativeDomainAuditError(
            "Evaluation denominator mismatch: "
            f"full_expected={len(expected_ids)}, "
            f"full_actual={full_metrics.get('total')!r}, "
            f"heldout_expected={len(heldout_ids)}, "
            f"heldout_actual={heldout_metrics.get('total')!r}"
        )
    provenance: JsonObject = require_object(
        payload.get("provenance"),
        "evaluation.provenance",
    )
    answer_evidence: JsonObject = require_object(
        provenance.get("answers"),
        "evaluation.provenance.answers",
    )
    answer_sha: str = sha256_file(answers_path)
    if answer_evidence.get("sha256") != answer_sha:
        raise NativeDomainAuditError(
            "Evaluation answer SHA mismatch: "
            f"expected={answer_sha}, actual={answer_evidence.get('sha256')!r}"
        )
    return metrics, heldout_ids


def validate_usage_identity(
    events: Sequence[UsageEvent],
    expected_ids: set[str],
    served_model: str,
    job_id: str,
    context: str,
) -> dict[str, int]:
    """Validate usage identity and return event counts by instance."""

    counts: dict[str, int] = {}
    for event in events:
        instance_id: str = event["instance_id"]
        if instance_id not in expected_ids:
            raise NativeDomainAuditError(
                "Usage event is outside the expected support: "
                f"context={context}, instance_id={instance_id}"
            )
        if event["served_model"] != served_model or event["job_id"] != job_id:
            raise NativeDomainAuditError(
                "Usage event identity mismatch: "
                f"context={context}, instance_id={instance_id}, "
                f"expected_model={served_model}, "
                f"actual_model={event['served_model']}, "
                f"expected_job={job_id}, actual_job={event['job_id']}"
            )
        counts[instance_id] = counts.get(instance_id, 0) + 1
    return counts


def validate_decision_usage(
    spec: UsageLogSpec,
    expected_calls: Mapping[str, int],
    served_model: str,
    job_id: str,
) -> JsonObject:
    """Require exactly one attributed usage event per decision client call."""

    events: list[UsageEvent] = load_usage_events(spec)
    observed: dict[str, int] = validate_usage_identity(
        events,
        set(expected_calls),
        served_model,
        job_id,
        "decision-usage",
    )
    if observed != dict(expected_calls):
        raise NativeDomainAuditError(
            "Decision usage call coverage mismatch: "
            f"expected_sample={list(expected_calls.items())[:20]}, "
            f"observed_sample={list(observed.items())[:20]}"
        )
    return {
        "path": str(spec["path"]),
        "sha256": sha256_file(spec["path"]),
        **summarize_events(events),
        "all_client_calls_have_usage_events": True,
    }


def validate_answer_usage(
    spec: UsageLogSpec,
    minimum_calls: Mapping[str, int],
    served_model: str,
    job_id: str,
) -> JsonObject:
    """Require usage for every non-zero-call answer attempt."""

    events: list[UsageEvent] = load_usage_events(spec)
    expected_ids: set[str] = {
        instance_id
        for instance_id, calls in minimum_calls.items()
        if calls > 0
    }
    observed: dict[str, int] = validate_usage_identity(
        events,
        expected_ids,
        served_model,
        job_id,
        "answer-usage",
    )
    if set(observed) != expected_ids:
        raise NativeDomainAuditError(
            "Answer usage instance coverage mismatch: "
            f"missing={sorted(expected_ids - set(observed))[:20]}, "
            f"unexpected={sorted(set(observed) - expected_ids)[:20]}"
        )
    insufficient: list[str] = [
        instance_id
        for instance_id, calls in minimum_calls.items()
        if calls > 0 and observed.get(instance_id, 0) < calls
    ]
    if insufficient:
        raise NativeDomainAuditError(
            "Answer usage has fewer HTTP events than engine attempts: "
            f"sample={insufficient[:20]}"
        )
    return {
        "path": str(spec["path"]),
        "sha256": sha256_file(spec["path"]),
        **summarize_events(events),
        "nonzero_call_answers": len(expected_ids),
        "zero_call_answers": len(minimum_calls) - len(expected_ids),
        "all_nonzero_call_answers_have_usage_events": True,
    }


def load_full_summary(
    path: Path,
    expected_event: str,
    expected_count: int,
    expected_output_sha: str,
    expected_identity: Mapping[str, JsonValue],
) -> JsonObject:
    """Load and validate one terminal full-run summary event."""

    if not path.is_file():
        raise FileNotFoundError(f"Full-run log does not exist: path={path}")
    matching: list[JsonObject] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip() or not line.lstrip().startswith("{"):
            continue
        try:
            value: JsonValue = cast(JsonValue, json.loads(line))
        except json.JSONDecodeError as error:
            raise NativeDomainAuditError(
                "Full-run log contains malformed JSON: "
                f"path={path}, line={line_number}, message={error.msg}"
            ) from error
        row: JsonObject = require_object(
            value,
            f"full-log:{path}:{line_number}",
        )
        if row.get("event") == expected_event:
            matching.append(row)
    if len(matching) != 1:
        raise NativeDomainAuditError(
            "Expected exactly one terminal full summary: "
            f"path={path}, event={expected_event}, observed={len(matching)}"
        )
    summary: JsonObject = matching[0]
    identity_mismatches: list[str] = [
        f"{field_name}:expected={expected!r},"
        f"actual={summary.get(field_name)!r}"
        for field_name, expected in expected_identity.items()
        if summary.get(field_name) != expected
    ]
    if (
        identity_mismatches
        or summary.get("run_mode") != "full"
        or summary.get("expected") != expected_count
        or summary.get("observed") != expected_count
        or summary.get("output_sha256") != expected_output_sha
    ):
        raise NativeDomainAuditError(
            "Full-run summary identity or denominator mismatch: "
            f"path={path}, summary={summary}"
        )
    categories: JsonObject = require_object(
        summary.get("failure_categories"),
        f"full-log:{path}.failure_categories",
    )
    unresolved: int = sum(
        require_nonnegative_integer(
            categories.get(category, 0),
            f"full-log:{path}.failure_categories.{category}",
        )
        for category in ("infra_transient", "unclassified_error")
    )
    if unresolved:
        raise NativeDomainAuditError(
            f"Full-run summary contains unresolved outcomes: count={unresolved}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "event": expected_event,
        "failure_categories": categories,
    }


def combine_usage(
    decision_usage: Mapping[str, JsonValue],
    answer_usage: Mapping[str, JsonValue],
) -> JsonObject:
    """Sum actual decision and answer calls and reported tokens."""

    return {
        field_name: require_nonnegative_integer(
            decision_usage.get(field_name),
            f"decision_usage.{field_name}",
        )
        + require_nonnegative_integer(
            answer_usage.get(field_name),
            f"answer_usage.{field_name}",
        )
        for field_name in (
            "http_calls",
            "response_calls",
            "error_calls",
            "usage_reported_calls",
            "usage_missing_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }


def write_json_atomic(path: Path, payload: JsonObject) -> None:
    """Write one formatted JSON result atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    """Validate one completed native-domain job and emit its audit."""

    args = parse_args()
    instances_path: Path = cast(Path, args.instances).resolve()
    decisions_path: Path = cast(Path, args.decisions).resolve()
    answers_path: Path = cast(Path, args.answers).resolve()
    decision_attempt_path: Path = cast(
        Path,
        args.decision_attempt_log,
    ).resolve()
    answer_attempt_path: Path = cast(
        Path,
        args.answer_attempt_log,
    ).resolve()
    decision_full_log_path: Path = cast(
        Path,
        args.decision_full_log,
    ).resolve()
    answer_full_log_path: Path = cast(
        Path,
        args.answer_full_log,
    ).resolve()
    decision_manifest_path: Path = cast(
        Path,
        args.decision_manifest,
    ).resolve()
    answer_manifest_path: Path = cast(
        Path,
        args.answer_manifest,
    ).resolve()
    evaluation_path: Path = cast(Path, args.evaluation).resolve()
    repository_root: Path = cast(Path, args.repository_root).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    result_tag: str = str(args.result_tag)
    served_model: str = str(args.served_model)
    domain: str = str(args.domain)
    arm: NativeArm = cast(NativeArm, str(args.arm))
    expected_count: int = int(args.expected_count)

    instances: dict[str, JsonObject] = load_instances(
        instances_path,
        domain,
        expected_count,
    )
    expected_ids: set[str] = set(instances)
    (
        decision_manifest,
        decision_manifest_sha,
        decision_job_id,
    ) = validate_manifest(
        decision_manifest_path,
        repository_root,
        instances_path,
        None,
        result_tag,
        served_model,
        domain,
        arm,
        "decision",
    )
    (
        answer_manifest,
        answer_manifest_sha,
        answer_job_id,
    ) = validate_manifest(
        answer_manifest_path,
        repository_root,
        instances_path,
        decisions_path,
        result_tag,
        served_model,
        domain,
        arm,
        "answer",
    )
    decisions, decision_calls = validate_decisions(
        load_jsonl(decisions_path, "decisions"),
        expected_ids,
        result_tag,
        served_model,
        domain,
        arm,
        decision_manifest,
        decision_manifest_sha,
    )
    decision_sha: str = sha256_file(decisions_path)
    answers, answer_calls = validate_answers(
        load_jsonl(answers_path, "answers"),
        expected_ids,
        result_tag,
        served_model,
        domain,
        arm,
        decision_sha,
        answer_manifest,
        answer_manifest_sha,
    )
    evaluation_metrics, heldout_ids = validate_evaluation(
        evaluation_path,
        answers_path,
        expected_ids,
        answers,
        result_tag,
        served_model,
        domain,
        arm,
    )
    decision_usage: JsonObject = validate_decision_usage(
        {
            "model": result_tag,
            "domain": domain,
            "arm": arm,
            "stage": "decision",
            "path": decision_attempt_path,
        },
        decision_calls,
        served_model,
        decision_job_id,
    )
    answer_usage: JsonObject = validate_answer_usage(
        {
            "model": result_tag,
            "domain": domain,
            "arm": arm,
            "stage": "answer",
            "path": answer_attempt_path,
        },
        answer_calls,
        served_model,
        answer_job_id,
    )
    decision_event: str = (
        "runtime_matched_select_decisions_complete"
        if arm == "select_bm25"
        else "rerank_decision_complete"
    )
    answer_event: str = (
        "runtime_matched_select_answers_complete"
        if arm == "select_bm25"
        else "rerank_answer_complete"
    )
    payload: JsonObject = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "model": result_tag,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "expected_rows": expected_count,
        "observed_decisions": len(decisions),
        "observed_answers": len(answers),
        "failure_categories": {
            "decision": count_categories(decisions),
            "answer": count_categories(answers),
        },
        "selection": {
            "full": selection_metrics(
                instances,
                decisions,
                expected_ids,
            ),
            "heldout": selection_metrics(
                instances,
                decisions,
                heldout_ids,
            ),
        },
        "answer_metrics": evaluation_metrics,
        "usage": {
            "decision": decision_usage,
            "answer": answer_usage,
            "combined": combine_usage(
                decision_usage,
                answer_usage,
            ),
            "token_source": "actual OpenAI-compatible response usage",
            "missing_usage_is_never_imputed": True,
        },
        "artifacts": {
            "instances": {
                "path": str(instances_path),
                "sha256": sha256_file(instances_path),
            },
            "decisions": {
                "path": str(decisions_path),
                "sha256": decision_sha,
            },
            "answers": {
                "path": str(answers_path),
                "sha256": sha256_file(answers_path),
            },
            "evaluation": {
                "path": str(evaluation_path),
                "sha256": sha256_file(evaluation_path),
            },
            "decision_manifest": {
                "path": str(decision_manifest_path),
                "sha256": decision_manifest_sha,
                "code_bundle_sha256": decision_manifest[
                    "code_bundle_sha256"
                ],
            },
            "answer_manifest": {
                "path": str(answer_manifest_path),
                "sha256": answer_manifest_sha,
                "code_bundle_sha256": answer_manifest[
                    "code_bundle_sha256"
                ],
            },
            "decision_full_log": load_full_summary(
                decision_full_log_path,
                decision_event,
                expected_count,
                decision_sha,
                (
                    {"job_id": decision_job_id}
                    if arm == "select_bm25"
                    else {
                        "result_tag": result_tag,
                        "served_model": served_model,
                        "domain": domain,
                    }
                ),
            ),
            "answer_full_log": load_full_summary(
                answer_full_log_path,
                answer_event,
                expected_count,
                sha256_file(answers_path),
                (
                    {"job_id": answer_job_id}
                    if arm == "select_bm25"
                    else {
                        "result_tag": result_tag,
                        "served_model": served_model,
                        "domain": domain,
                    }
                ),
            ),
        },
        "fresh_only": True,
        "reused_same_arm": 0,
        "unresolved": 0,
        "valid": True,
    }
    write_json_atomic(output_path, payload)
    print(
        canonical_json(
            {
                "event": "runtime_matched_native_domain_audit_complete",
                "model": result_tag,
                "domain": domain,
                "arm": arm,
                "observed_decisions": len(decisions),
                "observed_answers": len(answers),
                "heldout_gold_load_rate": payload["selection"][
                    "heldout"
                ]["gold_load_rate"],
                "total_tokens": payload["usage"]["combined"][
                    "total_tokens"
                ],
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "valid": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
