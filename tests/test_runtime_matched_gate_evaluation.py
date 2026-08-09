"""Focused tests for mixed-schema changed-Gate evaluation."""

from __future__ import annotations

from typing import cast

import pytest

from hyskill.runtime_matched_execution import sha256_json
from hyskill.runtime_matched_gate import (
    GATE_MERGE_REPORT_SCHEMA_VERSION,
    GATE_RERUN_ANSWER_SCHEMA_VERSION,
    GateArm,
)
from scripts.evaluate_runtime_matched_gate_merge import (
    EVALUATION_ROW_SCHEMA_VERSION,
    OLD_ANSWER_SCHEMA_VERSION,
    GateMergeEvaluationError,
    JsonObject,
    JsonValue,
    MergeReportEvidence,
    evaluation_row,
    normalize_answer,
    validate_merge_report,
    validate_normalized_counts,
)


RESULT_TAG: str = "llama31-8b"
SERVED_MODEL: str = "llama31-8b"
DOMAIN: str = "theoremqa"
ARM: str = "routed_gated"
RUNTIME_MANIFEST_SHA256: str = "a" * 64
CODE_BUNDLE_SHA256: str = "b" * 64
MERGED_ANSWERS_SHA256: str = "c" * 64


def merge_report_fixture() -> JsonObject:
    """Return one internally signed two-row Gate merge report."""

    report: JsonObject = {
        "schema_version": GATE_MERGE_REPORT_SCHEMA_VERSION,
        "valid": True,
        "model": RESULT_TAG,
        "served_model": SERVED_MODEL,
        "domain": DOMAIN,
        "arm": ARM,
        "expected_count": 2,
        "observed_count": 2,
        "preserved_row_count": 1,
        "rerun_row_count": 1,
        "preserved_method_failure_count": 0,
        "rerun_success_count": 0,
        "rerun_method_failure_count": 1,
        "unresolved_outcome_count": 0,
        "unchanged_rows_preserved_byte_for_byte": True,
        "preserved_line_bundle_sha256": "d" * 64,
        "rerun_line_bundle_sha256": "e" * 64,
        "audit": {"path": "/producer/audit.json", "sha256": "f" * 64},
        "diff": {"path": "/producer/diff.jsonl", "sha256": "1" * 64},
        "old_answers": {
            "path": "/producer/old.jsonl",
            "sha256": "2" * 64,
        },
        "rerun_answers": {
            "path": "/producer/rerun.jsonl",
            "sha256": "3" * 64,
        },
        "runtime_manifest": {
            "path": "/producer/runtime.json",
            "sha256": RUNTIME_MANIFEST_SHA256,
            "code_bundle_sha256": CODE_BUNDLE_SHA256,
        },
        "merged_answers": {
            "path": "/producer/merged.jsonl",
            "sha256": MERGED_ANSWERS_SHA256,
        },
    }
    report["evidence_bundle_sha256"] = sha256_json(report)
    return report


def report_evidence_fixture() -> MergeReportEvidence:
    """Validate and return fixture merge evidence."""

    return validate_merge_report(
        merge_report_fixture(),
        MERGED_ANSWERS_SHA256,
        RESULT_TAG,
        SERVED_MODEL,
        DOMAIN,
        cast(GateArm, ARM),
        2,
    )


def old_answer_fixture() -> JsonObject:
    """Return one preserved formal K=2 answer row."""

    return {
        "schema_version": OLD_ANSWER_SCHEMA_VERSION,
        "instance_id": "theoremqa_old",
        "model": RESULT_TAG,
        "served_model": SERVED_MODEL,
        "dataset": DOMAIN,
        "method": ARM,
        "expected_skill_ids": ["skill_old"],
        "skill_ids_used": ["skill_old"],
        "actual_injection_state": {
            "state": "confirmed_by_engine",
            "skill_ids": ["skill_old"],
        },
        "raw_output": "The answer is 42.",
        "failure_category": "success",
        "request_hash": "4" * 64,
        "answer_code_bundle_sha256": "5" * 64,
        "runtime_identity": {
            "model": RESULT_TAG,
            "served_model": SERVED_MODEL,
        },
    }


def rerun_answer_fixture() -> JsonObject:
    """Return one fresh changed-payload method failure."""

    return {
        "schema_version": GATE_RERUN_ANSWER_SCHEMA_VERSION,
        "stage": "answer",
        "instance_id": "theoremqa_new",
        "model": RESULT_TAG,
        "served_model": SERVED_MODEL,
        "dataset": DOMAIN,
        "domain": DOMAIN,
        "method": ARM,
        "arm": ARM,
        "expected_skill_ids": [],
        "skill_ids_used": [],
        "actual_injection_state": {
            "state": "bare",
            "skill_ids": [],
        },
        "raw_output": "",
        "failure_category": "method_failure",
        "answer_payload_hash": "6" * 64,
        "execution_request_hash": "7" * 64,
        "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "code_bundle_sha256": CODE_BUNDLE_SHA256,
        "reused_same_arm": False,
    }


def test_mixed_answers_keep_failure_in_denominator() -> None:
    """Evaluate preserved success and fresh failure under one row schema."""

    report_evidence: MergeReportEvidence = report_evidence_fixture()
    old_answer = normalize_answer(
        old_answer_fixture(),
        "theoremqa_old",
        RESULT_TAG,
        SERVED_MODEL,
        DOMAIN,
        cast(GateArm, ARM),
        report_evidence,
    )
    rerun_answer = normalize_answer(
        rerun_answer_fixture(),
        "theoremqa_new",
        RESULT_TAG,
        SERVED_MODEL,
        DOMAIN,
        cast(GateArm, ARM),
        report_evidence,
    )
    schema_counts: JsonObject = validate_normalized_counts(
        [old_answer, rerun_answer],
        report_evidence,
    )
    evaluator_calls: list[str] = []

    def evaluator(
        raw_output: str,
        instance: dict[str, object],
    ) -> dict[str, object]:
        del instance
        evaluator_calls.append(raw_output)
        return {"correct": raw_output.endswith("42.")}

    old_row = evaluation_row(
        "theoremqa_old",
        {
            "instance_id": "theoremqa_old",
            "dataset": DOMAIN,
            "eval_data": {"answer": "42"},
        },
        old_answer,
        set(),
        evaluator,
        RESULT_TAG,
        SERVED_MODEL,
        DOMAIN,
        cast(GateArm, ARM),
    )
    new_row = evaluation_row(
        "theoremqa_new",
        {
            "instance_id": "theoremqa_new",
            "dataset": DOMAIN,
            "eval_data": {"answer": "42"},
        },
        rerun_answer,
        {"theoremqa_new"},
        evaluator,
        RESULT_TAG,
        SERVED_MODEL,
        DOMAIN,
        cast(GateArm, ARM),
    )
    assert schema_counts == {
        OLD_ANSWER_SCHEMA_VERSION: 1,
        GATE_RERUN_ANSWER_SCHEMA_VERSION: 1,
    }
    assert old_row["schema_version"] == EVALUATION_ROW_SCHEMA_VERSION
    assert old_row["correct"] is True
    assert old_row["request_identity_kind"] == "legacy_request_hash"
    assert new_row["correct"] is False
    assert new_row["is_validation"] is True
    assert new_row["request_identity_kind"] == "execution_request_hash"
    assert evaluator_calls == ["The answer is 42."]


def test_merge_report_rejects_tampering_and_count_drift() -> None:
    """Reject unsigned report edits and observed schema-count drift."""

    tampered_report: JsonObject = merge_report_fixture()
    tampered_report["rerun_success_count"] = 1
    with pytest.raises(
        GateMergeEvaluationError,
        match="evidence bundle SHA",
    ):
        validate_merge_report(
            tampered_report,
            MERGED_ANSWERS_SHA256,
            RESULT_TAG,
            SERVED_MODEL,
            DOMAIN,
            cast(GateArm, ARM),
            2,
        )

    evidence: MergeReportEvidence = report_evidence_fixture()
    old_answer = normalize_answer(
        old_answer_fixture(),
        "theoremqa_old",
        RESULT_TAG,
        SERVED_MODEL,
        DOMAIN,
        cast(GateArm, ARM),
        evidence,
    )
    with pytest.raises(
        GateMergeEvaluationError,
        match="schema/outcome counts",
    ):
        validate_normalized_counts(
            [old_answer, old_answer],
            evidence,
        )


def test_fresh_row_must_match_report_runtime_identity() -> None:
    """Reject a changed answer bound to another runtime manifest."""

    rerun: JsonObject = rerun_answer_fixture()
    rerun["runtime_manifest_sha256"] = "8" * 64
    with pytest.raises(
        GateMergeEvaluationError,
        match="runtime_manifest_sha256",
    ):
        normalize_answer(
            rerun,
            "theoremqa_new",
            RESULT_TAG,
            SERVED_MODEL,
            DOMAIN,
            cast(GateArm, ARM),
            report_evidence_fixture(),
        )


def test_qwen_reference_preserved_row_uses_runtime_model_tag() -> None:
    """Accept only the frozen legacy Qwen reference model-tag convention."""

    answer: JsonObject = old_answer_fixture()
    answer["model"] = "qwen3.5-4b"
    answer["served_model"] = "qwen3.5-4b"
    runtime_identity: JsonObject = cast(
        JsonObject,
        answer["runtime_identity"],
    )
    runtime_identity["model"] = "qwen3.5-4b-reference"
    runtime_identity["served_model"] = "qwen3.5-4b"
    normalized = normalize_answer(
        answer,
        "theoremqa_old",
        "qwen3.5-4b-reference",
        "qwen3.5-4b",
        DOMAIN,
        cast(GateArm, ARM),
        report_evidence_fixture(),
    )
    assert normalized["source_schema_version"] == OLD_ANSWER_SCHEMA_VERSION

    answer["model"] = "unexpected-model"
    with pytest.raises(
        GateMergeEvaluationError,
        match="answer identity mismatch",
    ):
        normalize_answer(
            answer,
            "theoremqa_old",
            "qwen3.5-4b-reference",
            "qwen3.5-4b",
            DOMAIN,
            cast(GateArm, ARM),
            report_evidence_fixture(),
        )
