#!/usr/bin/env python3
"""Evaluate one validated mixed-schema changed-Gate answer merge."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

from hyskill.runtime_matched_execution import (
    canonical_json,
    require_sha256,
    sha256_file,
    sha256_json,
    sha256_text,
)
from hyskill.runtime_matched_gate import (
    GATE_MERGE_REPORT_SCHEMA_VERSION,
    GATE_RERUN_ANSWER_SCHEMA_VERSION,
    GateArm,
)
from scripts.evaluate_runtime_matched_baselines import (
    BaselineEvaluationError,
    EvaluateOne,
    JsonObject,
    JsonValue,
    index_rows,
    load_instances,
    load_json,
    load_jsonl,
    load_native_evaluator,
    load_validation_ids,
    require_boolean,
    require_ground_truth,
    require_object,
    require_string,
    require_string_list,
    require_text,
    write_json_atomic,
)


OLD_ANSWER_SCHEMA_VERSION: str = "k2-answer-record-v1"
EVALUATION_ROW_SCHEMA_VERSION: str = "k2-answer-evaluation-row-v1"
EVALUATION_SCHEMA_VERSION: str = "k2-answer-evaluation-v1"
RESOLVED_CATEGORIES: frozenset[str] = frozenset(
    {
        "success",
        "method_failure",
    }
)

FailureCategory = Literal["success", "method_failure"]
RequestIdentityKind = Literal[
    "legacy_request_hash",
    "execution_request_hash",
]


class GateMergeEvaluationError(BaselineEvaluationError):
    """Raised when a changed-Gate merge cannot be evaluated strictly."""


class NormalizedAnswer(TypedDict):
    """Schema-independent answer evidence required by the evaluator."""

    source_schema_version: str
    request_identity_kind: RequestIdentityKind
    request_hash: str
    raw_output: str
    failure_category: FailureCategory
    expected_skill_ids: list[str]
    skill_ids_used: list[str]


class EvaluationRow(TypedDict):
    """One K=2-compatible evaluation row with source-schema provenance."""

    schema_version: str
    instance_id: str
    model: str
    served_model: str
    domain: str
    arm: GateArm
    correct: bool
    failure_category: FailureCategory
    request_hash: str
    request_identity_kind: RequestIdentityKind
    source_answer_schema_version: str
    expected_skill_ids: list[str]
    skill_ids_used: list[str]
    is_validation: bool
    ground_truth: JsonValue
    raw_output_sha256: str
    evaluator: JsonObject


class MergeReportEvidence(TypedDict):
    """Validated merge report facts required for row verification."""

    preserved_row_count: int
    rerun_row_count: int
    preserved_method_failure_count: int
    rerun_success_count: int
    rerun_method_failure_count: int
    runtime_manifest_sha256: str
    code_bundle_sha256: str
    evidence_bundle_sha256: str


def expected_preserved_answer_model(
    result_tag: str,
    served_model: str,
) -> str:
    """Return the frozen legacy answer-row model tag."""

    if (
        result_tag == "qwen3.5-4b-reference"
        and served_model == "qwen3.5-4b"
    ):
        return served_model
    return result_tag


def parse_args() -> argparse.Namespace:
    """Parse one explicit mixed-schema Gate evaluation job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--merge-report", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--validation-source", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("routed_gated", "fixed_gated"),
    )
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_nonnegative_integer(
    value: JsonValue | None,
    context: str,
) -> int:
    """Return one nonnegative JSON integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateMergeEvaluationError(
            "Expected nonnegative integer: "
            f"context={context}, value={value!r}"
        )
    return value


def require_gate_arm(value: JsonValue | None, context: str) -> GateArm:
    """Return one supported Gate arm."""

    arm: str = require_string(value, context)
    if arm not in ("routed_gated", "fixed_gated"):
        raise GateMergeEvaluationError(
            f"Unsupported Gate arm: context={context}, arm={arm}"
        )
    return cast(GateArm, arm)


def require_failure_category(
    value: JsonValue | None,
    context: str,
) -> FailureCategory:
    """Return one resolved answer outcome."""

    if not isinstance(value, str) or value not in RESOLVED_CATEGORIES:
        raise GateMergeEvaluationError(
            "Gate evaluation accepts only resolved answer outcomes: "
            f"context={context}, value={value!r}"
        )
    return cast(FailureCategory, value)


def require_report_artifact_sha256(
    report: Mapping[str, JsonValue],
    artifact_name: str,
) -> str:
    """Return one SHA-bound artifact from the merge report."""

    artifact: JsonObject = require_object(
        report.get(artifact_name),
        f"merge-report.{artifact_name}",
    )
    return require_sha256(
        artifact.get("sha256"),
        f"merge-report.{artifact_name}.sha256",
    )


def validate_merge_report(
    report: JsonObject,
    answers_sha256: str,
    result_tag: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    expected_count: int,
) -> MergeReportEvidence:
    """Validate report identity, arithmetic, and merged-answer binding."""

    if report.get("schema_version") != GATE_MERGE_REPORT_SCHEMA_VERSION:
        raise GateMergeEvaluationError(
            "Unexpected Gate merge report schema: "
            f"schema={report.get('schema_version')!r}"
        )
    if report.get("valid") is not True:
        raise GateMergeEvaluationError(
            f"Gate merge report is not valid: valid={report.get('valid')!r}"
        )
    evidence_bundle_sha256: str = require_sha256(
        report.get("evidence_bundle_sha256"),
        "merge-report.evidence_bundle_sha256",
    )
    unsigned_report: JsonObject = {
        key: value
        for key, value in report.items()
        if key != "evidence_bundle_sha256"
    }
    observed_bundle_sha256: str = sha256_json(unsigned_report)
    if observed_bundle_sha256 != evidence_bundle_sha256:
        raise GateMergeEvaluationError(
            "Gate merge evidence bundle SHA is invalid: "
            f"expected={evidence_bundle_sha256}, "
            f"observed={observed_bundle_sha256}"
        )
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("model", result_tag),
        ("served_model", served_model),
        ("domain", domain),
        ("arm", arm),
        ("expected_count", expected_count),
        ("observed_count", expected_count),
        ("unresolved_outcome_count", 0),
        ("unchanged_rows_preserved_byte_for_byte", True),
    )
    mismatches: list[str] = [
        f"{field}:expected={expected!r},observed={report.get(field)!r}"
        for field, expected in expected_fields
        if report.get(field) != expected
    ]
    if mismatches:
        raise GateMergeEvaluationError(
            f"Gate merge report identity mismatch: mismatches={mismatches}"
        )
    preserved_count: int = require_nonnegative_integer(
        report.get("preserved_row_count"),
        "merge-report.preserved_row_count",
    )
    rerun_count: int = require_nonnegative_integer(
        report.get("rerun_row_count"),
        "merge-report.rerun_row_count",
    )
    preserved_failures: int = require_nonnegative_integer(
        report.get("preserved_method_failure_count"),
        "merge-report.preserved_method_failure_count",
    )
    rerun_successes: int = require_nonnegative_integer(
        report.get("rerun_success_count"),
        "merge-report.rerun_success_count",
    )
    rerun_failures: int = require_nonnegative_integer(
        report.get("rerun_method_failure_count"),
        "merge-report.rerun_method_failure_count",
    )
    if preserved_count <= 0 or rerun_count <= 0:
        raise GateMergeEvaluationError(
            "Mixed-schema evaluation requires both preserved and rerun rows: "
            f"preserved={preserved_count}, rerun={rerun_count}"
        )
    arithmetic_mismatches: list[str] = []
    if preserved_count + rerun_count != expected_count:
        arithmetic_mismatches.append(
            "preserved+rerun:"
            f"expected={expected_count},"
            f"observed={preserved_count + rerun_count}"
        )
    if rerun_successes + rerun_failures != rerun_count:
        arithmetic_mismatches.append(
            "rerun-success+failure:"
            f"expected={rerun_count},"
            f"observed={rerun_successes + rerun_failures}"
        )
    if preserved_failures > preserved_count:
        arithmetic_mismatches.append(
            "preserved-method-failure:"
            f"preserved={preserved_count},failures={preserved_failures}"
        )
    if arithmetic_mismatches:
        raise GateMergeEvaluationError(
            "Gate merge report count mismatch: "
            f"mismatches={arithmetic_mismatches}"
        )
    declared_answers_sha256: str = require_report_artifact_sha256(
        report,
        "merged_answers",
    )
    if declared_answers_sha256 != answers_sha256:
        raise GateMergeEvaluationError(
            "Merged answer SHA differs from its merge report: "
            f"expected={declared_answers_sha256}, observed={answers_sha256}"
        )
    for artifact_name in (
        "audit",
        "diff",
        "old_answers",
        "rerun_answers",
    ):
        require_report_artifact_sha256(report, artifact_name)
    runtime_manifest: JsonObject = require_object(
        report.get("runtime_manifest"),
        "merge-report.runtime_manifest",
    )
    runtime_manifest_sha256: str = require_sha256(
        runtime_manifest.get("sha256"),
        "merge-report.runtime_manifest.sha256",
    )
    code_bundle_sha256: str = require_sha256(
        runtime_manifest.get("code_bundle_sha256"),
        "merge-report.runtime_manifest.code_bundle_sha256",
    )
    return {
        "preserved_row_count": preserved_count,
        "rerun_row_count": rerun_count,
        "preserved_method_failure_count": preserved_failures,
        "rerun_success_count": rerun_successes,
        "rerun_method_failure_count": rerun_failures,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "code_bundle_sha256": code_bundle_sha256,
        "evidence_bundle_sha256": evidence_bundle_sha256,
    }


def validate_answer_identity(
    answer: Mapping[str, JsonValue],
    instance_id: str,
    answer_model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
) -> None:
    """Validate fields shared by preserved and fresh Gate answers."""

    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("instance_id", instance_id),
        ("model", answer_model),
        ("served_model", served_model),
        ("dataset", domain),
        ("method", arm),
    )
    mismatches: list[str] = [
        f"{field}:expected={expected!r},observed={answer.get(field)!r}"
        for field, expected in expected_fields
        if answer.get(field) != expected
    ]
    if mismatches:
        raise GateMergeEvaluationError(
            "Merged Gate answer identity mismatch: "
            f"instance_id={instance_id}, mismatches={mismatches}"
        )


def validate_injection_and_outcome(
    answer: Mapping[str, JsonValue],
    instance_id: str,
) -> tuple[str, FailureCategory, list[str], list[str]]:
    """Validate raw output, loaded skills, and resolved outcome."""

    raw_output: str = require_text(
        answer.get("raw_output"),
        f"answers:{instance_id}.raw_output",
    )
    expected_skill_ids: list[str] = require_string_list(
        answer.get("expected_skill_ids"),
        f"answers:{instance_id}.expected_skill_ids",
    )
    skill_ids_used: list[str] = require_string_list(
        answer.get("skill_ids_used"),
        f"answers:{instance_id}.skill_ids_used",
    )
    injection_state: JsonObject = require_object(
        answer.get("actual_injection_state"),
        f"answers:{instance_id}.actual_injection_state",
    )
    injected_skill_ids: list[str] = require_string_list(
        injection_state.get("skill_ids"),
        f"answers:{instance_id}.actual_injection_state.skill_ids",
    )
    if injected_skill_ids != expected_skill_ids:
        raise GateMergeEvaluationError(
            "Merged Gate answer submitted unexpected skills: "
            f"instance_id={instance_id}, expected={expected_skill_ids}, "
            f"observed={injected_skill_ids}"
        )
    category: FailureCategory = require_failure_category(
        answer.get("failure_category"),
        f"answers:{instance_id}.failure_category",
    )
    if category == "success":
        if not raw_output.strip() or skill_ids_used != expected_skill_ids:
            raise GateMergeEvaluationError(
                "Successful merged Gate answer has invalid output or skills: "
                f"instance_id={instance_id}"
            )
    elif raw_output or skill_ids_used:
        raise GateMergeEvaluationError(
            "Merged Gate method failure carries output or used skills: "
            f"instance_id={instance_id}"
        )
    return raw_output, category, expected_skill_ids, skill_ids_used


def normalize_old_answer(
    answer: JsonObject,
    instance_id: str,
    result_tag: str,
    served_model: str,
    domain: str,
    arm: GateArm,
) -> NormalizedAnswer:
    """Normalize one byte-preserved formal K=2 answer."""

    validate_answer_identity(
        answer,
        instance_id,
        expected_preserved_answer_model(result_tag, served_model),
        served_model,
        domain,
        arm,
    )
    raw_output, category, expected_skills, used_skills = (
        validate_injection_and_outcome(answer, instance_id)
    )
    runtime_identity: JsonObject = require_object(
        answer.get("runtime_identity"),
        f"answers:{instance_id}.runtime_identity",
    )
    if (
        runtime_identity.get("model") != result_tag
        or runtime_identity.get("served_model") != served_model
    ):
        raise GateMergeEvaluationError(
            "Preserved K=2 runtime identity mismatch: "
            f"instance_id={instance_id}, runtime_identity={runtime_identity}"
        )
    require_sha256(
        answer.get("answer_code_bundle_sha256"),
        f"answers:{instance_id}.answer_code_bundle_sha256",
    )
    request_hash: str = require_sha256(
        answer.get("request_hash"),
        f"answers:{instance_id}.request_hash",
    )
    return {
        "source_schema_version": OLD_ANSWER_SCHEMA_VERSION,
        "request_identity_kind": "legacy_request_hash",
        "request_hash": request_hash,
        "raw_output": raw_output,
        "failure_category": category,
        "expected_skill_ids": expected_skills,
        "skill_ids_used": used_skills,
    }


def normalize_rerun_answer(
    answer: JsonObject,
    instance_id: str,
    result_tag: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    report_evidence: MergeReportEvidence,
) -> NormalizedAnswer:
    """Normalize one fresh runtime-matched changed-payload answer."""

    validate_answer_identity(
        answer,
        instance_id,
        result_tag,
        served_model,
        domain,
        arm,
    )
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("domain", domain),
        ("arm", arm),
        ("stage", "answer"),
        ("reused_same_arm", False),
        (
            "runtime_manifest_sha256",
            report_evidence["runtime_manifest_sha256"],
        ),
        (
            "code_bundle_sha256",
            report_evidence["code_bundle_sha256"],
        ),
    )
    mismatches: list[str] = [
        f"{field}:expected={expected!r},observed={answer.get(field)!r}"
        for field, expected in expected_fields
        if answer.get(field) != expected
    ]
    if mismatches:
        raise GateMergeEvaluationError(
            "Fresh changed-Gate answer identity mismatch: "
            f"instance_id={instance_id}, mismatches={mismatches}"
        )
    raw_output, category, expected_skills, used_skills = (
        validate_injection_and_outcome(answer, instance_id)
    )
    require_sha256(
        answer.get("answer_payload_hash"),
        f"answers:{instance_id}.answer_payload_hash",
    )
    execution_hash: str = require_sha256(
        answer.get("execution_request_hash"),
        f"answers:{instance_id}.execution_request_hash",
    )
    return {
        "source_schema_version": GATE_RERUN_ANSWER_SCHEMA_VERSION,
        "request_identity_kind": "execution_request_hash",
        "request_hash": execution_hash,
        "raw_output": raw_output,
        "failure_category": category,
        "expected_skill_ids": expected_skills,
        "skill_ids_used": used_skills,
    }


def normalize_answer(
    answer: JsonObject,
    instance_id: str,
    result_tag: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    report_evidence: MergeReportEvidence,
) -> NormalizedAnswer:
    """Dispatch one merged answer by its explicit source schema."""

    schema_version: JsonValue | None = answer.get("schema_version")
    if schema_version == OLD_ANSWER_SCHEMA_VERSION:
        return normalize_old_answer(
            answer,
            instance_id,
            result_tag,
            served_model,
            domain,
            arm,
        )
    if schema_version == GATE_RERUN_ANSWER_SCHEMA_VERSION:
        return normalize_rerun_answer(
            answer,
            instance_id,
            result_tag,
            served_model,
            domain,
            arm,
            report_evidence,
        )
    raise GateMergeEvaluationError(
        "Merged Gate answer uses an unsupported schema: "
        f"instance_id={instance_id}, schema={schema_version!r}"
    )


def validate_normalized_counts(
    answers: Sequence[NormalizedAnswer],
    report_evidence: MergeReportEvidence,
) -> JsonObject:
    """Match source-schema and outcome counts to the merge report."""

    preserved: list[NormalizedAnswer] = [
        answer
        for answer in answers
        if answer["source_schema_version"] == OLD_ANSWER_SCHEMA_VERSION
    ]
    rerun: list[NormalizedAnswer] = [
        answer
        for answer in answers
        if answer["source_schema_version"]
        == GATE_RERUN_ANSWER_SCHEMA_VERSION
    ]
    observed_counts: dict[str, int] = {
        "preserved_row_count": len(preserved),
        "rerun_row_count": len(rerun),
        "preserved_method_failure_count": sum(
            answer["failure_category"] == "method_failure"
            for answer in preserved
        ),
        "rerun_success_count": sum(
            answer["failure_category"] == "success"
            for answer in rerun
        ),
        "rerun_method_failure_count": sum(
            answer["failure_category"] == "method_failure"
            for answer in rerun
        ),
    }
    expected_counts: dict[str, int] = {
        "preserved_row_count": report_evidence["preserved_row_count"],
        "rerun_row_count": report_evidence["rerun_row_count"],
        "preserved_method_failure_count": report_evidence[
            "preserved_method_failure_count"
        ],
        "rerun_success_count": report_evidence["rerun_success_count"],
        "rerun_method_failure_count": report_evidence[
            "rerun_method_failure_count"
        ],
    }
    mismatches: list[str] = [
        f"{name}:expected={expected_counts[name]},"
        f"observed={observed_counts[name]}"
        for name in observed_counts
        if observed_counts[name] != expected_counts[name]
    ]
    if mismatches:
        raise GateMergeEvaluationError(
            "Merged answer schema/outcome counts differ from the report: "
            f"mismatches={mismatches}"
        )
    return {
        OLD_ANSWER_SCHEMA_VERSION: len(preserved),
        GATE_RERUN_ANSWER_SCHEMA_VERSION: len(rerun),
    }


def evaluate_success(
    evaluator: EvaluateOne,
    raw_output: str,
    instance: JsonObject,
) -> JsonObject:
    """Evaluate one successful answer with the frozen native scorer."""

    raw_result: dict[str, object] = evaluator(
        raw_output,
        cast(dict[str, object], instance),
    )
    result: JsonObject = cast(JsonObject, raw_result)
    require_boolean(result.get("correct"), "native-evaluator.correct")
    return result


def evaluation_row(
    instance_id: str,
    instance: JsonObject,
    answer: NormalizedAnswer,
    validation_ids: set[str],
    evaluator: EvaluateOne,
    result_tag: str,
    served_model: str,
    domain: str,
    arm: GateArm,
) -> EvaluationRow:
    """Build one K=2-compatible evaluation row."""

    if answer["failure_category"] == "success":
        evaluator_result: JsonObject = evaluate_success(
            evaluator,
            answer["raw_output"],
            instance,
        )
    else:
        evaluator_result = {
            "correct": False,
            "extracted_answer": "",
            "evaluation_status": "method_failure",
        }
    return {
        "schema_version": EVALUATION_ROW_SCHEMA_VERSION,
        "instance_id": instance_id,
        "model": result_tag,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "correct": require_boolean(
            evaluator_result.get("correct"),
            f"evaluator:{instance_id}.correct",
        ),
        "failure_category": answer["failure_category"],
        "request_hash": answer["request_hash"],
        "request_identity_kind": answer["request_identity_kind"],
        "source_answer_schema_version": answer["source_schema_version"],
        "expected_skill_ids": answer["expected_skill_ids"],
        "skill_ids_used": answer["skill_ids_used"],
        "is_validation": instance_id in validation_ids,
        "ground_truth": require_ground_truth(instance),
        "raw_output_sha256": sha256_text(answer["raw_output"]),
        "evaluator": evaluator_result,
    }


def metric_summary(rows: Sequence[EvaluationRow]) -> JsonObject:
    """Compute accuracy and resolved outcome counts."""

    if not rows:
        raise GateMergeEvaluationError(
            "Cannot summarize an empty Gate evaluation support"
        )
    categories: dict[str, int] = {}
    for row in rows:
        category: str = row["failure_category"]
        categories[category] = categories.get(category, 0) + 1
    correct: int = sum(row["correct"] for row in rows)
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "failure_categories": categories,
    }


def main() -> None:
    """Evaluate one complete mixed-schema changed-Gate merge."""

    args = parse_args()
    answers_path: Path = cast(Path, args.answers).resolve()
    merge_report_path: Path = cast(Path, args.merge_report).resolve()
    instances_path: Path = cast(Path, args.instances).resolve()
    validation_source_path: Path = cast(
        Path,
        args.validation_source,
    ).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    result_tag: str = str(args.result_tag)
    served_model: str = str(args.served_model)
    domain: str = str(args.domain)
    arm: GateArm = require_gate_arm(str(args.arm), "args.arm")
    expected_count: int = int(args.expected_count)
    if expected_count <= 0:
        raise ValueError(
            f"expected-count must be positive: value={expected_count}"
        )
    input_paths: set[Path] = {
        answers_path,
        merge_report_path,
        instances_path,
        validation_source_path,
    }
    if output_path in input_paths:
        raise GateMergeEvaluationError(
            "Gate evaluation must not overwrite an input artifact: "
            f"path={output_path}"
        )

    instances: dict[str, JsonObject] = load_instances(
        instances_path,
        domain,
    )
    answers: dict[str, JsonObject] = index_rows(
        load_jsonl(answers_path, "merged answers"),
        "merged answers",
    )
    expected_ids: set[str] = set(instances)
    observed_ids: set[str] = set(answers)
    if expected_ids != observed_ids:
        raise GateMergeEvaluationError(
            "Merged Gate answer coverage mismatch: "
            f"missing={sorted(expected_ids - observed_ids)[:20]}, "
            f"unexpected={sorted(observed_ids - expected_ids)[:20]}"
        )
    if len(instances) != expected_count:
        raise GateMergeEvaluationError(
            "Gate evaluation denominator mismatch: "
            f"expected={expected_count}, actual={len(instances)}"
        )
    raw_report: JsonValue = load_json(
        merge_report_path,
        "merge report",
    )
    report: JsonObject = require_object(raw_report, "merge report")
    report_evidence: MergeReportEvidence = validate_merge_report(
        report,
        sha256_file(answers_path),
        result_tag,
        served_model,
        domain,
        arm,
        expected_count,
    )
    normalized_by_id: dict[str, NormalizedAnswer] = {
        instance_id: normalize_answer(
            answers[instance_id],
            instance_id,
            result_tag,
            served_model,
            domain,
            arm,
            report_evidence,
        )
        for instance_id in instances
    }
    schema_counts: JsonObject = validate_normalized_counts(
        list(normalized_by_id.values()),
        report_evidence,
    )
    validation_ids: set[str] = load_validation_ids(
        validation_source_path
    )
    unexpected_validation_ids: list[str] = sorted(
        validation_ids - expected_ids
    )
    if unexpected_validation_ids:
        raise GateMergeEvaluationError(
            "Validation IDs are outside the instance support: "
            f"sample={unexpected_validation_ids[:20]}"
        )
    evaluator: EvaluateOne = load_native_evaluator()
    rows: list[EvaluationRow] = [
        evaluation_row(
            instance_id,
            instances[instance_id],
            normalized_by_id[instance_id],
            validation_ids,
            evaluator,
            result_tag,
            served_model,
            domain,
            arm,
        )
        for instance_id in instances
    ]
    heldout_rows: list[EvaluationRow] = [
        row for row in rows if not row["is_validation"]
    ]
    payload: JsonObject = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "model": result_tag,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "metrics": {
            "full": metric_summary(rows),
            "heldout": metric_summary(heldout_rows),
        },
        "provenance": {
            "answers": {
                "path": str(answers_path),
                "sha256": sha256_file(answers_path),
            },
            "merge_report": {
                "path": str(merge_report_path),
                "sha256": sha256_file(merge_report_path),
                "evidence_bundle_sha256": report_evidence[
                    "evidence_bundle_sha256"
                ],
            },
            "instances": {
                "path": str(instances_path),
                "sha256": sha256_file(instances_path),
            },
            "validation_source": {
                "path": str(validation_source_path),
                "sha256": sha256_file(validation_source_path),
            },
            "evaluation_code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "answer_schema_counts": schema_counts,
            "request_hash_semantics": {
                OLD_ANSWER_SCHEMA_VERSION: "legacy_request_hash",
                GATE_RERUN_ANSWER_SCHEMA_VERSION: (
                    "execution_request_hash"
                ),
            },
            "heldout_authority": "frozen K=2 validation-source val_ids",
            "legacy_compact_baseline_read": False,
        },
        "details": cast(list[JsonValue], rows),
    }
    write_json_atomic(output_path, payload)
    print(
        canonical_json(
            {
                "event": "runtime_matched_gate_evaluation_complete",
                "model": result_tag,
                "domain": domain,
                "arm": arm,
                "expected": expected_count,
                "full": payload["metrics"]["full"],
                "heldout": payload["metrics"]["heldout"],
                "schema_counts": schema_counts,
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
