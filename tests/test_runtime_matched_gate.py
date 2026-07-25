"""Focused regression tests for runtime-matched Gate recalibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from hyskill.runtime_matched_execution import (
    FROZEN_K2_RUNTIME_REFERENCES,
    execution_request_hash,
)
from hyskill.runtime_matched_gate import (
    ANSWER_PAYLOAD_SCHEMA_VERSION,
    GATE_AUDIT_SCHEMA_VERSION,
    GATE_RERUN_ANSWER_SCHEMA_VERSION,
    RULE_DOMAIN_COUNTS,
    GateAuditResult,
    GateSignal,
    JsonObject,
    NativeGateRuntime,
    RuntimeMatchedGateError,
    audit_gate_task,
    calibrate_thresholds,
    expected_gate_row_count,
    expected_gate_task_keys,
    expected_skill_ids,
    gate_decision,
    pick_tau,
    render_direct_answer_payload,
)
from scripts.merge_runtime_matched_gate_answers import (
    RawJsonlLine,
    merge_gate_answer_lines,
)
from scripts.summarize_runtime_matched_gate import (
    GateTaskArtifacts,
    combine_gate_tasks,
    require_expected_task_matrix,
    validate_diff_rows,
)


def test_pick_tau_reproduces_strict_less_than_boundary() -> None:
    values: list[float] = [0.1, 0.2, 0.3, 0.4]
    labels: list[bool] = [True, True, False, True]
    assert pick_tau(values, labels, 0.9) == 0.2


def test_fresh_bare_labels_can_change_only_tau2() -> None:
    signals: list[GateSignal] = [
        {
            "instance_id": "a",
            "top1": "s1",
            "S1": 0.1,
            "S2": 0.1,
            "rel_truth_wrong": True,
        },
        {
            "instance_id": "b",
            "top1": "s2",
            "S1": 0.2,
            "S2": 0.2,
            "rel_truth_wrong": False,
        },
    ]
    thresholds = calibrate_thresholds(
        signals,
        frozenset({"a", "b"}),
        {"a": True, "b": False},
        0.9,
    )
    assert thresholds == {"tau1": 0.1, "tau2": 0.1}
    assert gate_decision(signals[0], thresholds["tau1"], thresholds["tau2"]) == "kept"
    assert expected_skill_ids(signals[0], "kept") == ("s1",)
    assert expected_skill_ids(signals[0], "blocked_s1") == ()


def test_rendered_payload_changes_when_skill_injection_changes() -> None:
    instance = {
        "instance_id": "example",
        "dataset": "logicbench",
        "question": "Q",
    }

    def build_prompt(
        current_instance: dict[str, object],
        skills: list[str] | None,
    ) -> tuple[str, str]:
        prefix: str = "" if skills is None else skills[0] + "\n"
        return "", prefix + str(current_instance["question"])

    def get_extra_body(
        served_model: str,
        thinking: bool,
    ) -> dict[str, object] | None:
        assert served_model == "model"
        assert thinking is False
        return {"chat_template_kwargs": {"enable_thinking": False}}

    bare_payload = render_direct_answer_payload(
        instance,
        [],
        "model",
        build_prompt,
        get_extra_body,
    )
    skill_payload = render_direct_answer_payload(
        instance,
        [{"skill_id": "s1", "content": "Skill"}],
        "model",
        build_prompt,
        get_extra_body,
    )
    assert bare_payload["answer_payload_hash"] != skill_payload["answer_payload_hash"]
    assert bare_payload["tools"] == []
    assert skill_payload["loaded_skills"][0]["skill_id"] == "s1"


def _old_answer(
    instance_id: str,
    expected_skills: list[str],
    failure_category: str,
    request_hash: str,
) -> JsonObject:
    raw_output: str = "old answer" if failure_category == "success" else ""
    skill_ids_used: list[str] = (
        list(expected_skills) if failure_category == "success" else []
    )
    return {
        "schema_version": "k2-answer-record-v1",
        "instance_id": instance_id,
        "dataset": "theoremqa",
        "method": "routed_gated",
        "model": "deepseek7b",
        "served_model": "deepseek7b",
        "expected_skill_ids": list(expected_skills),
        "skill_ids_used": skill_ids_used,
        "actual_injection_state": {
            "state": (
                "confirmed_by_engine"
                if failure_category == "success"
                else "request_submitted"
            ),
            "skill_ids": list(expected_skills),
        },
        "failure_category": failure_category,
        "raw_output": raw_output,
        "request_hash": request_hash,
        "runtime_identity": {
            "model": "deepseek7b",
            "served_model": "deepseek7b",
        },
    }


def _gate_runtime() -> NativeGateRuntime:
    def build_prompt(
        instance: dict[str, object],
        skills: list[str] | None,
    ) -> tuple[str, str]:
        prefix: str = "" if skills is None else skills[0] + "\n"
        return "System", prefix + str(instance["question"])

    return {
        "build_prompt": build_prompt,
        "get_extra_body": lambda model, thinking: {
            "served_model": model,
            "enable_thinking": thinking,
        },
        "revision": "revision",
        "source_root": "/source",
    }


def _gate_audit_fixture() -> tuple[
    GateAuditResult,
    dict[str, JsonObject],
]:
    instances: dict[str, JsonObject] = {
        instance_id: {
            "instance_id": instance_id,
            "dataset": "theoremqa",
            "question": f"Question {instance_id}",
        }
        for instance_id in ("cal_a", "cal_b", "changed")
    }
    corpus: dict[str, JsonObject] = {
        "s1": {
            "skill_id": "s1",
            "content": "Skill content",
        }
    }
    signals: dict[str, GateSignal] = {
        "cal_a": {
            "instance_id": "cal_a",
            "top1": "s1",
            "S1": 0.1,
            "S2": 0.1,
            "rel_truth_wrong": True,
        },
        "cal_b": {
            "instance_id": "cal_b",
            "top1": "s1",
            "S1": 0.2,
            "S2": 0.2,
            "rel_truth_wrong": True,
        },
        "changed": {
            "instance_id": "changed",
            "top1": "s1",
            "S1": 0.3,
            "S2": 0.15,
            "rel_truth_wrong": False,
        },
    }
    old_answers: dict[str, JsonObject] = {
        "cal_a": _old_answer("cal_a", [], "method_failure", "a" * 64),
        "cal_b": _old_answer("cal_b", ["s1"], "success", "b" * 64),
        "changed": _old_answer(
            "changed",
            ["s1"],
            "success",
            "c" * 64,
        ),
    }
    result: GateAuditResult = audit_gate_task(
        instances,
        corpus,
        signals,
        frozenset({"cal_a", "cal_b"}),
        {"cal_a": True, "cal_b": True, "changed": True},
        {"tau1": 0.2, "tau2": 0.1},
        {"cal_a": [], "cal_b": ["s1"], "changed": ["s1"]},
        old_answers,
        "deepseek7b",
        "deepseek7b",
        "theoremqa",
        "routed_gated",
        0.9,
        _gate_runtime(),
    )
    return result, old_answers


def test_gate_audit_reruns_only_changed_injection_payload() -> None:
    result, _ = _gate_audit_fixture()
    diff_by_id: dict[str, JsonObject] = {
        cast(str, row["instance_id"]): row for row in result["diff_rows"]
    }
    assert result["new_thresholds"] == {"tau1": 0.2, "tau2": 0.2}
    assert diff_by_id["changed"]["decision_changed"] is True
    assert diff_by_id["changed"]["injection_changed"] is True
    assert diff_by_id["changed"]["payload_changed"] is True
    assert diff_by_id["changed"]["rerun_required"] is True
    assert diff_by_id["cal_a"]["old_failure_category"] == "method_failure"
    assert diff_by_id["cal_a"]["preserve_old_row"] is True
    assert diff_by_id["cal_a"]["rerun_required"] is False


def test_changed_decision_with_same_injection_does_not_require_rerun() -> None:
    payload_hash: str = "d" * 64
    row: JsonObject = {
        "schema_version": "runtime-matched-gate-diff-row-v1",
        "model": "deepseek7b",
        "served_model": "deepseek7b",
        "domain": "theoremqa",
        "arm": "routed_gated",
        "instance_id": "same-payload",
        "is_validation": False,
        "S1": 0.1,
        "S2": 0.1,
        "top1_skill_id": "s1",
        "old_tau1": 0.2,
        "old_tau2": None,
        "new_tau1": 0.2,
        "new_tau2": 0.2,
        "old_decision": "blocked_s1",
        "new_decision": "skipped_s2",
        "old_expected_skill_ids": [],
        "new_expected_skill_ids": [],
        "decision_changed": True,
        "injection_changed": False,
        "old_answer_payload_hash": payload_hash,
        "new_answer_payload_hash": payload_hash,
        "payload_changed": False,
        "rerun_required": False,
        "preserve_old_row": True,
        "old_request_hash": "e" * 64,
        "old_failure_category": "success",
    }
    indexed: dict[str, JsonObject] = validate_diff_rows(
        [row],
        ("deepseek7b", "theoremqa", "routed_gated"),
        "deepseek7b",
        1,
        {"tau1": 0.2, "tau2": None},
        {"tau1": 0.2, "tau2": 0.2},
    )
    assert indexed["same-payload"]["rerun_required"] is False


def _raw_line(
    line_number: int,
    row: JsonObject,
    compact: bool,
) -> RawJsonlLine:
    separators: tuple[str, str] | None = (",", ":") if compact else None
    raw_bytes: bytes = (
        json.dumps(
            row,
            ensure_ascii=False,
            separators=separators,
        )
        + "\n"
    ).encode("utf-8")
    return {
        "line_number": line_number,
        "raw_bytes": raw_bytes,
        "record": row,
    }


def _rerun_answer(
    diff: JsonObject,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
) -> JsonObject:
    payload_hash: str = cast(str, diff["new_answer_payload_hash"])
    expected_skills: list[str] = cast(
        list[str],
        diff["new_expected_skill_ids"],
    )
    return {
        "schema_version": GATE_RERUN_ANSWER_SCHEMA_VERSION,
        "instance_id": diff["instance_id"],
        "model": "deepseek7b",
        "served_model": "deepseek7b",
        "domain": "theoremqa",
        "arm": "routed_gated",
        "stage": "answer",
        "raw_output": "fresh changed answer",
        "expected_skill_ids": expected_skills,
        "skill_ids_used": expected_skills,
        "actual_injection_state": {
            "state": "confirmed_by_engine",
            "skill_ids": expected_skills,
        },
        "failure_category": "success",
        "answer_payload_hash": payload_hash,
        "execution_request_hash": execution_request_hash(
            ANSWER_PAYLOAD_SCHEMA_VERSION,
            payload_hash,
            runtime_manifest_sha256,
            code_bundle_sha256,
        ),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "code_bundle_sha256": code_bundle_sha256,
        "reused_same_arm": False,
    }


def test_changed_row_merge_preserves_old_bytes_and_method_failure() -> None:
    result, old_answers = _gate_audit_fixture()
    old_lines: list[RawJsonlLine] = [
        _raw_line(index, old_answers[instance_id], False)
        for index, instance_id in enumerate(
            ("cal_a", "cal_b", "changed"),
            start=1,
        )
    ]
    diff_by_id: dict[str, JsonObject] = {
        cast(str, row["instance_id"]): row for row in result["diff_rows"]
    }
    runtime_manifest_sha256: str = "f" * 64
    code_bundle_sha256: str = "1" * 64
    rerun_line: RawJsonlLine = _raw_line(
        1,
        _rerun_answer(
            diff_by_id["changed"],
            runtime_manifest_sha256,
            code_bundle_sha256,
        ),
        True,
    )
    merged = merge_gate_answer_lines(
        old_lines,
        [rerun_line],
        result["diff_rows"],
        "deepseek7b",
        "deepseek7b",
        "theoremqa",
        "routed_gated",
        runtime_manifest_sha256,
        code_bundle_sha256,
    )
    assert merged["preserved_count"] == 2
    assert merged["rerun_count"] == 1
    assert merged["preserved_method_failure_count"] == 1
    assert merged["output_lines"][0] == old_lines[0]["raw_bytes"]
    assert merged["output_lines"][1] == old_lines[1]["raw_bytes"]
    assert merged["output_lines"][2] == rerun_line["raw_bytes"]


def test_changed_row_merge_rejects_missing_or_extra_rerun_rows() -> None:
    result, old_answers = _gate_audit_fixture()
    old_lines: list[RawJsonlLine] = [
        _raw_line(index, old_answers[instance_id], True)
        for index, instance_id in enumerate(
            ("cal_a", "cal_b", "changed"),
            start=1,
        )
    ]
    with pytest.raises(
        RuntimeMatchedGateError,
        match="exactly the rerun-required IDs",
    ):
        merge_gate_answer_lines(
            old_lines,
            [],
            result["diff_rows"],
            "deepseek7b",
            "deepseek7b",
            "theoremqa",
            "routed_gated",
            "f" * 64,
            "1" * 64,
        )
    extra_row: JsonObject = {
        "instance_id": "extra",
    }
    with pytest.raises(
        RuntimeMatchedGateError,
        match="exactly the rerun-required IDs",
    ):
        merge_gate_answer_lines(
            old_lines,
            [_raw_line(1, extra_row, True)],
            result["diff_rows"],
            "deepseek7b",
            "deepseek7b",
            "theoremqa",
            "routed_gated",
            "f" * 64,
            "1" * 64,
        )


def _aggregate_task_fixture(
    key: tuple[str, str, str],
) -> GateTaskArtifacts:
    model, domain, raw_arm = key
    arm = cast(str, raw_arm)
    count: int = RULE_DOMAIN_COUNTS[domain]
    rows: list[JsonObject] = [
        {"instance_id": f"{domain}_{index:05d}"}
        for index in range(count)
    ]
    audit: JsonObject = {
        "schema_version": GATE_AUDIT_SCHEMA_VERSION,
        "decision_change_count": 0,
        "injection_change_count": 0,
        "payload_change_count": 0,
        "preserved_method_failure_count": 0,
    }
    return cast(
        GateTaskArtifacts,
        {
            "key": (model, domain, arm),
            "model": model,
            "served_model": FROZEN_K2_RUNTIME_REFERENCES[model][
                "served_model"
            ],
            "domain": domain,
            "arm": arm,
            "expected_count": count,
            "audit_path": Path(f"/audit/{model}-{domain}-{arm}.json"),
            "audit_sha256": "2" * 64,
            "audit": audit,
            "diff_path": Path(f"/diff/{model}-{domain}-{arm}.jsonl"),
            "diff_sha256": "3" * 64,
            "diff_rows": rows,
            "decision_path": Path(
                f"/decision/{model}-{domain}-{arm}.jsonl"
            ),
            "decision_sha256": "4" * 64,
            "rerun_path": Path(f"/rerun/{model}-{domain}-{arm}.json"),
            "rerun_sha256": "5" * 64,
            "rerun_rows": [],
        },
    )


def test_exact_32_task_gate_aggregate_support() -> None:
    keys = expected_gate_task_keys()
    assert len(keys) == 32
    assert sum(key[2] == "routed_gated" for key in keys) == 28
    assert sum(key[2] == "fixed_gated" for key in keys) == 4
    require_expected_task_matrix(keys)
    tasks: list[GateTaskArtifacts] = [
        _aggregate_task_fixture(cast(tuple[str, str, str], key))
        for key in keys
    ]
    combined_rows, manifest = combine_gate_tasks(tasks)
    assert len(combined_rows) == expected_gate_row_count() == 22640
    assert manifest["observed_task_count"] == 32
    assert manifest["rerun_required_count"] == 0
    with pytest.raises(RuntimeMatchedGateError, match="task matrix mismatch"):
        require_expected_task_matrix(keys[:-1])
