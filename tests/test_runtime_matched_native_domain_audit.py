"""Focused tests for native-domain completeness and selection metrics."""

from __future__ import annotations

from typing import cast

import pytest

from hyskill.runtime_matched_execution import JobBoundManifest
from scripts.audit_runtime_matched_native_domain import (
    NativeDomainAuditError,
    selection_metrics,
    validate_answers,
    validate_usage_identity,
)
from scripts.summarize_runtime_matched_usage import (
    JsonObject,
    UsageEvent,
)


def candidate_ids(first: str, second: str) -> list[str]:
    """Return 50 unique candidate IDs with two explicit leaders."""

    return [first, second, *[f"skill_{index:02d}" for index in range(48)]]


def test_selection_metrics_include_fallbacks_and_gold_annotations() -> None:
    """Treat deterministic fallback as loaded and report all three metrics."""

    instances: dict[str, JsonObject] = {
        "theoremqa_00000": {
            "instance_id": "theoremqa_00000",
            "dataset": "theoremqa",
            "skill_annotations": ["gold_a"],
        },
        "theoremqa_00001": {
            "instance_id": "theoremqa_00001",
            "dataset": "theoremqa",
            "skill_annotations": ["gold_b"],
        },
    }
    decisions: dict[str, JsonObject] = {
        "theoremqa_00000": {
            "instance_id": "theoremqa_00000",
            "failure_category": "success",
            "selected_skill_id": "gold_a",
            "ordered_candidate_ids": candidate_ids("gold_a", "other_a"),
        },
        "theoremqa_00001": {
            "instance_id": "theoremqa_00001",
            "failure_category": "selector_fallback",
            "selected_skill_id": "other_b",
            "ordered_candidate_ids": candidate_ids("other_b", "gold_b"),
        },
    }
    metrics: JsonObject = selection_metrics(
        instances,
        decisions,
        set(instances),
    )
    assert metrics["loaded"] == 2
    assert metrics["gold_loaded"] == 1
    assert metrics["loaded_skill_precision"] == 0.5
    assert metrics["loading_rate"] == 1.0
    assert metrics["gold_load_rate"] == 0.5
    assert metrics["candidate_recall_at_50"] == 1.0
    assert metrics["bm25_rank1_gold_rate"] == 0.5


def answer_row(reused: bool) -> JsonObject:
    """Return one manifest-bound Select answer."""

    return {
        "schema_version": "runtime-matched-baseline-answer-v1",
        "instance_id": "theoremqa_00000",
        "model": "glm4-9b",
        "served_model": "glm4-9b",
        "domain": "theoremqa",
        "arm": "select_bm25",
        "stage": "answer",
        "runtime_manifest_sha256": "a" * 64,
        "code_bundle_sha256": "b" * 64,
        "decision_source_sha256": "c" * 64,
        "reused_same_arm": reused,
        "failure_category": "success",
        "answer_call_attempts": 1,
    }


def answer_manifest() -> JobBoundManifest:
    """Return the manifest field required by answer-row validation."""

    return cast(
        JobBoundManifest,
        {
            "code_bundle_sha256": "b" * 64,
        },
    )


def test_answer_validation_binds_decision_sha_and_rejects_reuse() -> None:
    """Reject stale decision bindings and reused answers."""

    rows, minimum_calls = validate_answers(
        [answer_row(False)],
        {"theoremqa_00000"},
        "glm4-9b",
        "glm4-9b",
        "theoremqa",
        "select_bm25",
        "c" * 64,
        answer_manifest(),
        "a" * 64,
    )
    assert list(rows) == ["theoremqa_00000"]
    assert minimum_calls == {"theoremqa_00000": 1}

    with pytest.raises(NativeDomainAuditError, match="reused_same_arm"):
        validate_answers(
            [answer_row(True)],
            {"theoremqa_00000"},
            "glm4-9b",
            "glm4-9b",
            "theoremqa",
            "select_bm25",
            "c" * 64,
            answer_manifest(),
            "a" * 64,
        )

    stale: JsonObject = answer_row(False)
    stale["decision_source_sha256"] = "d" * 64
    with pytest.raises(
        NativeDomainAuditError,
        match="decision_source_sha256",
    ):
        validate_answers(
            [stale],
            {"theoremqa_00000"},
            "glm4-9b",
            "glm4-9b",
            "theoremqa",
            "select_bm25",
            "c" * 64,
            answer_manifest(),
            "a" * 64,
        )


def usage_event(job_id: str) -> UsageEvent:
    """Return one attributed usage event."""

    return {
        "served_model": "glm4-9b",
        "job_id": job_id,
        "domain": "theoremqa",
        "arm": "select_bm25",
        "instance_id": "theoremqa_00000",
        "logical_attempt": 1,
        "http_subcall": 1,
        "status": "response",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "usage_missing_reason": None,
        "elapsed_seconds": 0.5,
    }


def test_usage_identity_is_job_and_model_bound() -> None:
    """Reject usage from another runtime-bound job."""

    assert validate_usage_identity(
        [usage_event("job-a")],
        {"theoremqa_00000"},
        "glm4-9b",
        "job-a",
        "fixture",
    ) == {"theoremqa_00000": 1}
    with pytest.raises(NativeDomainAuditError, match="identity mismatch"):
        validate_usage_identity(
            [usage_event("job-b")],
            {"theoremqa_00000"},
            "glm4-9b",
            "job-a",
            "fixture",
        )
