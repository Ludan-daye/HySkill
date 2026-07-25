#!/usr/bin/env python3
"""Recalibrate and diff one fresh-Bare-dependent K=2 Gate task."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO, cast

from hyskill.runtime_matched_execution import (
    FROZEN_K2_RUNTIME_REFERENCES,
    FROZEN_SRAGENTS_REVISION,
    JsonValue,
    canonical_json,
    sha256_file,
    sha256_json,
)
from hyskill.runtime_matched_gate import (
    FROZEN_CORPUS_SHA256,
    FROZEN_INSTANCE_SHA256,
    GATE_AUDIT_SCHEMA_VERSION,
    GATE_RERUN_ANSWER_SCHEMA_VERSION,
    GATE_RERUN_MANIFEST_SCHEMA_VERSION,
    P_MIN,
    GateArm,
    GateAuditResult,
    GateSignal,
    GateThresholds,
    JsonObject,
    NativeGateRuntime,
    RuntimeMatchedGateError,
    audit_gate_task,
    index_corpus,
    index_instances,
    index_rows,
    index_signals,
    load_native_gate_runtime,
    require_boolean,
    require_gate_arm,
    require_list,
    require_number,
    require_object,
    require_optional_number,
    require_string,
    require_string_list,
)


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-domain-arm Gate audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--signals", required=True, type=Path)
    parser.add_argument("--old-taus", required=True, type=Path)
    parser.add_argument("--old-gated", required=True, type=Path)
    parser.add_argument("--fresh-bare-eval", required=True, type=Path)
    parser.add_argument("--old-answers", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("routed_gated", "fixed_gated"),
    )
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--p-min", required=True, type=float)
    parser.add_argument("--sragents-checkout", required=True, type=Path)
    parser.add_argument(
        "--sragents-revision",
        required=True,
        choices=(FROZEN_SRAGENTS_REVISION,),
    )
    parser.add_argument("--diff-output", required=True, type=Path)
    parser.add_argument("--decision-output", required=True, type=Path)
    parser.add_argument("--rerun-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    return parser.parse_args()


def open_text(path: Path) -> TextIO:
    """Open one plain or gzip-compressed UTF-8 input."""

    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_json(path: Path, context: str) -> JsonValue:
    """Load one required JSON artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    try:
        with open_text(path) as input_file:
            return cast(JsonValue, json.load(input_file))
    except json.JSONDecodeError as error:
        raise RuntimeMatchedGateError(
            f"{context} JSON is malformed: path={path}, "
            f"line={error.lineno}, column={error.colno}, message={error.msg}"
        ) from error


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load one non-empty strict JSONL artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    rows: list[JsonObject] = []
    with open_text(path) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise RuntimeMatchedGateError(
                    f"{context} contains a blank JSONL line: "
                    f"path={path}, line={line_number}"
                )
            try:
                value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeMatchedGateError(
                    f"{context} JSONL is malformed: path={path}, "
                    f"line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            rows.append(
                require_object(
                    value,
                    f"{context}:{path}:{line_number}",
                )
            )
    if not rows:
        raise RuntimeMatchedGateError(
            f"{context} JSONL is empty: path={path}"
        )
    return rows


def write_json_atomic(path: Path, payload: Mapping[str, JsonValue]) -> None:
    """Atomically write one formatted JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, JsonValue]],
) -> None:
    """Atomically write canonical JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        for row in rows:
            output_file.write(canonical_json(row))
            output_file.write("\n")
    temporary_path.replace(path)


def load_thresholds(
    path: Path,
    p_min: float,
) -> tuple[GateThresholds, frozenset[str]]:
    """Load old thresholds and their frozen validation membership."""

    payload: JsonObject = require_object(
        load_json(path, "old-taus"),
        "old-taus",
    )
    observed_p_min: float = require_number(
        payload.get("p_min"),
        "old-taus.p_min",
    )
    if observed_p_min != p_min:
        raise RuntimeMatchedGateError(
            "Old Gate p_min differs from the frozen audit value: "
            f"expected={p_min}, observed={observed_p_min}"
        )
    validation_ids_list: list[str] = require_string_list(
        payload.get("val_ids"),
        "old-taus.val_ids",
    )
    validation_ids: frozenset[str] = frozenset(validation_ids_list)
    if len(validation_ids) != len(validation_ids_list):
        raise RuntimeMatchedGateError("old-taus.val_ids contains duplicates")
    raw_n_val: JsonValue | None = payload.get("n_val")
    if (
        isinstance(raw_n_val, bool)
        or not isinstance(raw_n_val, int)
        or raw_n_val != len(validation_ids)
    ):
        raise RuntimeMatchedGateError(
            "Old Gate n_val differs from val_ids: "
            f"n_val={raw_n_val!r}, val_ids={len(validation_ids)}"
        )
    return (
        {
            "tau1": require_optional_number(
                payload.get("tau1"),
                "old-taus.tau1",
            ),
            "tau2": require_optional_number(
                payload.get("tau2"),
                "old-taus.tau2",
            ),
        },
        validation_ids,
    )


def load_signal_payload(path: Path) -> dict[str, GateSignal]:
    """Load complete zero-cache-miss Gate signals."""

    payload: JsonObject = require_object(
        load_json(path, "signals"),
        "signals",
    )
    cache_misses: JsonValue | None = payload.get("cache_misses")
    if cache_misses != 0:
        raise RuntimeMatchedGateError(
            "Gate signals must have zero cache misses: "
            f"value={cache_misses!r}"
        )
    raw_signals: list[JsonValue] = require_list(
        payload.get("signals"),
        "signals.signals",
    )
    return index_signals(raw_signals)


def load_old_gated(path: Path) -> dict[str, list[str]]:
    """Load old gated retrieval decisions as ordered skill-ID lists."""

    payload: JsonObject = require_object(
        load_json(path, "old-gated"),
        "old-gated",
    )
    rows: dict[str, JsonObject] = index_rows(
        require_list(payload.get("results"), "old-gated.results"),
        "old-gated.results",
    )
    output: dict[str, list[str]] = {}
    for instance_id, row in rows.items():
        retrieved: list[JsonValue] = require_list(
            row.get("retrieved"),
            f"old-gated:{instance_id}.retrieved",
        )
        output[instance_id] = [
            require_string(
                require_object(
                    candidate,
                    f"old-gated:{instance_id}.retrieved[{index}]",
                ).get("skill_id"),
                f"old-gated:{instance_id}.retrieved[{index}].skill_id",
            )
            for index, candidate in enumerate(retrieved)
        ]
    return output


def load_fresh_bare_correctness(
    path: Path,
    result_tag: str,
    served_model: str,
    domain: str,
    validation_ids: frozenset[str],
) -> dict[str, bool]:
    """Load fresh Bare correctness with exact job and split identity."""

    payload: JsonObject = require_object(
        load_json(path, "fresh-bare-eval"),
        "fresh-bare-eval",
    )
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("schema_version", "runtime-matched-baseline-evaluation-v1"),
        ("model", result_tag),
        ("served_model", served_model),
        ("domain", domain),
        ("arm", "bare"),
    )
    mismatches: list[str] = [
        f"{field_name}:expected={expected!r},"
        f"observed={payload.get(field_name)!r}"
        for field_name, expected in expected_fields
        if payload.get(field_name) != expected
    ]
    if mismatches:
        raise RuntimeMatchedGateError(
            "Fresh Bare evaluation identity mismatch: "
            f"mismatches={mismatches}"
        )
    rows: dict[str, JsonObject] = index_rows(
        require_list(payload.get("details"), "fresh-bare-eval.details"),
        "fresh-bare-eval.details",
    )
    output: dict[str, bool] = {}
    for instance_id, row in rows.items():
        if row.get("model") != result_tag or row.get("domain") != domain:
            raise RuntimeMatchedGateError(
                "Fresh Bare detail identity mismatch: "
                f"instance_id={instance_id}"
            )
        if row.get("arm") != "bare":
            raise RuntimeMatchedGateError(
                "Fresh Bare detail uses a non-Bare arm: "
                f"instance_id={instance_id}, arm={row.get('arm')!r}"
            )
        expected_validation: bool = instance_id in validation_ids
        observed_validation: bool = require_boolean(
            row.get("is_validation"),
            f"fresh-bare-eval:{instance_id}.is_validation",
        )
        if observed_validation != expected_validation:
            raise RuntimeMatchedGateError(
                "Fresh Bare split membership differs from old Gate val_ids: "
                f"instance_id={instance_id}, "
                f"expected={expected_validation}, "
                f"observed={observed_validation}"
            )
        correct: bool = require_boolean(
            row.get("correct"),
            f"fresh-bare-eval:{instance_id}.correct",
        )
        category: str = require_string(
            row.get("failure_category"),
            f"fresh-bare-eval:{instance_id}.failure_category",
        )
        if category not in ("success", "method_failure"):
            raise RuntimeMatchedGateError(
                "Fresh Bare evaluation contains an unresolved outcome: "
                f"instance_id={instance_id}, category={category}"
            )
        if category == "method_failure" and correct:
            raise RuntimeMatchedGateError(
                "Fresh Bare method failure cannot be correct: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = correct
    return output


def validate_model_arm(
    result_tag: str,
    served_model: str,
    arm: GateArm,
) -> None:
    """Require one frozen K=2 model and supported Gate arm."""

    reference = FROZEN_K2_RUNTIME_REFERENCES.get(result_tag)
    if reference is None:
        raise RuntimeMatchedGateError(
            f"Unknown K=2 result tag: result_tag={result_tag!r}"
        )
    if reference["served_model"] != served_model:
        raise RuntimeMatchedGateError(
            "Served model differs from the frozen K=2 identity: "
            f"result_tag={result_tag}, expected={reference['served_model']}, "
            f"observed={served_model}"
        )
    if arm == "fixed_gated" and result_tag != "qwen3.5-4b-reference":
        raise RuntimeMatchedGateError(
            "Fixed Gate is supported only for Qwen3.5-4B reference: "
            f"result_tag={result_tag}"
        )


def require_frozen_input_hashes(
    instances_path: Path,
    corpus_path: Path,
    domain: str,
) -> None:
    """Bind the Gate audit to the exact K=2 corpus and instances."""

    expected_instances_sha256: str | None = FROZEN_INSTANCE_SHA256.get(domain)
    if expected_instances_sha256 is None:
        raise RuntimeMatchedGateError(
            f"Unknown rule domain for Gate audit: domain={domain!r}"
        )
    observed_instances_sha256: str = sha256_file(instances_path)
    observed_corpus_sha256: str = sha256_file(corpus_path)
    if observed_instances_sha256 != expected_instances_sha256:
        raise RuntimeMatchedGateError(
            "Instances SHA differs from the frozen K=2 input: "
            f"domain={domain}, expected={expected_instances_sha256}, "
            f"observed={observed_instances_sha256}"
        )
    if observed_corpus_sha256 != FROZEN_CORPUS_SHA256:
        raise RuntimeMatchedGateError(
            "Corpus SHA differs from the frozen K=2 input: "
            f"expected={FROZEN_CORPUS_SHA256}, "
            f"observed={observed_corpus_sha256}"
        )


def category_counts(
    rows: Sequence[Mapping[str, JsonValue]],
    field_name: str,
) -> dict[str, int]:
    """Count one required string field."""

    counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        value: str = require_string(
            row.get(field_name),
            f"rows[{index}].{field_name}",
        )
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> None:
    """Produce one complete Gate decision and answer-payload audit."""

    args = parse_args()
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    signals_path: Path = cast(Path, args.signals).resolve()
    old_taus_path: Path = cast(Path, args.old_taus).resolve()
    old_gated_path: Path = cast(Path, args.old_gated).resolve()
    fresh_bare_eval_path: Path = cast(Path, args.fresh_bare_eval).resolve()
    old_answers_path: Path = cast(Path, args.old_answers).resolve()
    checkout_path: Path = cast(Path, args.sragents_checkout).resolve()
    diff_output_path: Path = cast(Path, args.diff_output).resolve()
    decision_output_path: Path = cast(Path, args.decision_output).resolve()
    rerun_output_path: Path = cast(Path, args.rerun_output).resolve()
    audit_output_path: Path = cast(Path, args.audit_output).resolve()
    result_tag: str = str(args.result_tag)
    served_model: str = str(args.served_model)
    domain: str = str(args.domain)
    arm: GateArm = require_gate_arm(str(args.arm))
    expected_count: int = int(args.expected_count)
    p_min: float = float(args.p_min)
    revision: str = str(args.sragents_revision)
    if expected_count <= 0:
        raise ValueError(
            f"expected-count must be positive: value={expected_count}"
        )
    if p_min != P_MIN:
        raise ValueError(
            f"p-min must equal the frozen value {P_MIN}: value={p_min}"
        )
    output_paths: tuple[Path, ...] = (
        diff_output_path,
        decision_output_path,
        rerun_output_path,
        audit_output_path,
    )
    if len(set(output_paths)) != len(output_paths):
        raise RuntimeMatchedGateError(
            f"Gate audit outputs must be distinct: paths={output_paths}"
        )
    input_paths: tuple[Path, ...] = (
        instances_path,
        corpus_path,
        signals_path,
        old_taus_path,
        old_gated_path,
        fresh_bare_eval_path,
        old_answers_path,
    )
    if set(output_paths) & set(input_paths):
        raise RuntimeMatchedGateError(
            "Gate audit must not overwrite any input artifact: "
            f"aliases={sorted(set(output_paths) & set(input_paths))}"
        )
    validate_model_arm(result_tag, served_model, arm)
    require_frozen_input_hashes(instances_path, corpus_path, domain)
    old_thresholds, validation_ids = load_thresholds(old_taus_path, p_min)
    instance_values: list[JsonValue] = require_list(
        load_json(instances_path, "instances"),
        "instances",
    )
    corpus_values: list[JsonValue] = require_list(
        load_json(corpus_path, "corpus"),
        "corpus",
    )
    instances: dict[str, JsonObject] = index_instances(
        instance_values,
        domain,
    )
    corpus: dict[str, JsonObject] = index_corpus(corpus_values)
    if len(instances) != expected_count:
        raise RuntimeMatchedGateError(
            "Gate denominator mismatch: "
            f"expected={expected_count}, observed={len(instances)}"
        )
    signals: dict[str, GateSignal] = load_signal_payload(signals_path)
    old_gated: dict[str, list[str]] = load_old_gated(old_gated_path)
    fresh_bare_correct: dict[str, bool] = load_fresh_bare_correctness(
        fresh_bare_eval_path,
        result_tag,
        served_model,
        domain,
        validation_ids,
    )
    old_answers: dict[str, JsonObject] = index_rows(
        cast(list[JsonValue], load_jsonl(old_answers_path, "old-answers")),
        "old-answers",
    )
    runtime: NativeGateRuntime = load_native_gate_runtime(
        checkout_path,
        revision,
    )
    result: GateAuditResult = audit_gate_task(
        instances,
        corpus,
        signals,
        validation_ids,
        fresh_bare_correct,
        old_thresholds,
        old_gated,
        old_answers,
        result_tag,
        served_model,
        domain,
        arm,
        p_min,
        runtime,
    )
    write_jsonl_atomic(diff_output_path, result["diff_rows"])
    write_jsonl_atomic(decision_output_path, result["decision_rows"])
    changed_rows: list[JsonObject] = [
        {
            "instance_id": row["instance_id"],
            "model": result_tag,
            "served_model": served_model,
            "domain": domain,
            "arm": arm,
            "new_expected_skill_ids": row["new_expected_skill_ids"],
            "new_answer_payload_hash": row["new_answer_payload_hash"],
            "old_request_hash": row["old_request_hash"],
            "old_failure_category": row["old_failure_category"],
        }
        for row in result["diff_rows"]
        if row["rerun_required"] is True
    ]
    input_artifacts: JsonObject = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in (
            ("instances", instances_path),
            ("corpus", corpus_path),
            ("signals", signals_path),
            ("old_taus", old_taus_path),
            ("old_gated", old_gated_path),
            ("fresh_bare_eval", fresh_bare_eval_path),
            ("old_answers", old_answers_path),
        )
    }
    rerun_manifest: JsonObject = {
        "schema_version": GATE_RERUN_MANIFEST_SCHEMA_VERSION,
        "answer_schema_version": GATE_RERUN_ANSWER_SCHEMA_VERSION,
        "model": result_tag,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "old_thresholds": old_thresholds,
        "new_thresholds": result["new_thresholds"],
        "rerun_count": len(changed_rows),
        "rows": changed_rows,
        "required_answer_fields": [
            "schema_version",
            "instance_id",
            "model",
            "served_model",
            "domain",
            "arm",
            "stage",
            "raw_output",
            "skill_ids_used",
            "expected_skill_ids",
            "failure_category",
            "answer_payload_hash",
            "execution_request_hash",
            "runtime_manifest_sha256",
            "code_bundle_sha256",
            "actual_injection_state",
            "reused_same_arm",
        ],
        "input_artifacts": input_artifacts,
    }
    write_json_atomic(rerun_output_path, rerun_manifest)
    payload_change_count: int = len(changed_rows)
    decision_change_count: int = sum(
        row["decision_changed"] is True for row in result["diff_rows"]
    )
    injection_change_count: int = sum(
        row["injection_changed"] is True for row in result["diff_rows"]
    )
    preserved_method_failures: int = sum(
        row["rerun_required"] is False
        and row["old_failure_category"] == "method_failure"
        for row in result["diff_rows"]
    )
    code_files: list[JsonObject] = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for path in (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1]
            / "hyskill"
            / "runtime_matched_gate.py",
        )
    ]
    output_artifacts: JsonObject = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in (
            ("diff", diff_output_path),
            ("decisions", decision_output_path),
            ("rerun", rerun_output_path),
        )
    }
    audit: JsonObject = {
        "schema_version": GATE_AUDIT_SCHEMA_VERSION,
        "valid": True,
        "model": result_tag,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "expected_count": expected_count,
        "validation_count": len(validation_ids),
        "p_min": p_min,
        "old_thresholds": old_thresholds,
        "new_thresholds": result["new_thresholds"],
        "tau1_unchanged": (
            old_thresholds["tau1"] == result["new_thresholds"]["tau1"]
        ),
        "tau2_unchanged": (
            old_thresholds["tau2"] == result["new_thresholds"]["tau2"]
        ),
        "old_decision_counts": category_counts(
            result["diff_rows"],
            "old_decision",
        ),
        "new_decision_counts": category_counts(
            result["diff_rows"],
            "new_decision",
        ),
        "decision_change_count": decision_change_count,
        "injection_change_count": injection_change_count,
        "payload_change_count": payload_change_count,
        "rerun_required_count": payload_change_count,
        "preserved_row_count": expected_count - payload_change_count,
        "preserved_method_failure_count": preserved_method_failures,
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
        "sragents": {
            "revision": runtime["revision"],
            "source_root": runtime["source_root"],
        },
        "code_files": code_files,
        "code_bundle_sha256": sha256_json(code_files),
    }
    write_json_atomic(audit_output_path, audit)
    print(
        canonical_json(
            {
                "event": "runtime_matched_gate_audit_complete",
                "model": result_tag,
                "domain": domain,
                "arm": arm,
                "expected": expected_count,
                "decision_changes": decision_change_count,
                "payload_changes": payload_change_count,
                "rerun_required": payload_change_count,
                "audit_output": str(audit_output_path),
                "audit_sha256": sha256_file(audit_output_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
