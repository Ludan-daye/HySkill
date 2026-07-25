#!/usr/bin/env python3
"""Evaluate one validated K=2 answer job with the native SR-Agents scorer."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    FailureCategory,
    JsonObject,
    JsonValue,
    canonical_json,
    sha256_file,
    sha256_text,
    validate_failure_category,
)
from scripts.audit_k2_reuse import (
    require_list,
    require_object,
    require_string,
)


class EvaluateOne(Protocol):
    """Native SR-Agents evaluator dispatch contract."""

    def __call__(
        self,
        raw_output: str,
        instance: dict[str, object],
    ) -> dict[str, object]:
        """Evaluate one model output against one benchmark instance."""


def parse_args() -> argparse.Namespace:
    """Parse explicit inputs for one model-domain-arm evaluation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--validation-source", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=(
            "routed_always",
            "routed_gated",
            "routed_select",
            "fixed_gated",
        ),
    )
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path, context: str) -> JsonValue:
    """Load one JSON file with explicit parse context."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    try:
        return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise DownstreamDataError(
            f"{context} JSON is malformed: path={path}, "
            f"line={error.lineno}, column={error.colno}, message={error.msg}"
        ) from error


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load one non-empty JSONL file."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise DownstreamDataError(
                    f"{context} JSONL is malformed: path={path}, "
                    f"line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            rows.append(
                require_object(
                    raw_value,
                    f"{context}:{path}:{line_number}",
                )
            )
    if not rows:
        raise DownstreamDataError(f"{context} JSONL is empty: path={path}")
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
            raise DownstreamDataError(
                f"{context} contains duplicate instance ID: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def load_instances(path: Path, domain: str) -> dict[str, JsonObject]:
    """Load and validate one benchmark instance set."""

    raw_value: JsonValue = load_json(path, "instances")
    raw_rows: list[JsonValue] = require_list(raw_value, "instances")
    rows: list[JsonObject] = [
        require_object(row, f"instances:{index}")
        for index, row in enumerate(raw_rows)
    ]
    output: dict[str, JsonObject] = index_rows(rows, "instances")
    for instance_id, instance in output.items():
        actual_domain: str = require_string(
            instance.get("dataset"),
            f"instances:{instance_id}.dataset",
        )
        if actual_domain != domain:
            raise DownstreamDataError(
                "Instance dataset mismatch: "
                f"instance_id={instance_id}, expected={domain}, "
                f"actual={actual_domain}"
            )
    return output


def load_validation_ids(path: Path) -> set[str]:
    """Load the frozen calibration IDs used to define held-out support."""

    raw_value: JsonValue = load_json(path, "validation-source")
    payload: JsonObject = require_object(raw_value, "validation-source")
    raw_ids: list[JsonValue] = require_list(
        payload.get("val_ids"),
        "validation-source.val_ids",
    )
    validation_ids: list[str] = [
        require_string(value, f"validation-source.val_ids[{index}]")
        for index, value in enumerate(raw_ids)
    ]
    if len(validation_ids) != len(set(validation_ids)):
        raise DownstreamDataError(
            "Validation source contains duplicate IDs: "
            f"path={path}, count={len(validation_ids)}"
        )
    if not validation_ids:
        raise DownstreamDataError(
            f"Validation source contains no IDs: path={path}"
        )
    return set(validation_ids)


def load_native_evaluator() -> EvaluateOne:
    """Load the exact SR-Agents evaluator dispatcher."""

    try:
        evaluate_module: ModuleType = importlib.import_module(
            "sragents.evaluate"
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SR-Agents evaluator is unavailable. Add "
            "external/SR-Agents/src to PYTHONPATH before evaluation"
        ) from error
    evaluator: object = getattr(evaluate_module, "evaluate", None)
    if not callable(evaluator):
        raise RuntimeError(
            "SR-Agents evaluator dispatcher is not callable: "
            "module=sragents.evaluate, attribute=evaluate"
        )
    return cast(EvaluateOne, evaluator)


def normalize_string_list(value: JsonValue | None, context: str) -> list[str]:
    """Return one JSON string list."""

    raw_values: list[JsonValue] = require_list(value, context)
    return [
        require_string(item, f"{context}[{index}]")
        for index, item in enumerate(raw_values)
    ]


def require_text(value: JsonValue | None, context: str) -> str:
    """Return one JSON string, including an explicitly empty string."""

    if not isinstance(value, str):
        raise DownstreamDataError(
            f"Expected string: context={context}, value={value!r}"
        )
    return value


def require_audit_status(row: Mapping[str, JsonValue], instance_id: str) -> str:
    """Return one valid reuse audit status."""

    status: str = require_string(
        row.get("status"),
        f"audit:{instance_id}.status",
    )
    if status not in ("reused_same_arm", "needs_inference", "rejected"):
        raise DownstreamDataError(
            f"Unknown reuse audit status: instance_id={instance_id}, "
            f"status={status}"
        )
    return status


def answer_failure_category(
    answer: Mapping[str, JsonValue],
    audit_status: str,
    instance_id: str,
) -> FailureCategory:
    """Resolve the answer failure category after validator-approved reuse."""

    if audit_status == "reused_same_arm":
        raw_output: str = require_text(
            answer.get("raw_output"),
            f"answers:{instance_id}.raw_output",
        )
        if not raw_output.strip():
            raise DownstreamDataError(
                "Reused success answer is empty: "
                f"instance_id={instance_id}"
            )
        return "success"
    category: FailureCategory = validate_failure_category(
        answer.get("failure_category")
    )
    if category in ("infra_transient", "unclassified_error"):
        raise DownstreamDataError(
            "Evaluation cannot include unresolved answer failures: "
            f"instance_id={instance_id}, category={category}"
        )
    if category == "selector_fallback":
        raise DownstreamDataError(
            "Selector fallback is not an answer failure category: "
            f"instance_id={instance_id}"
        )
    return category


def require_ground_truth(instance: Mapping[str, JsonValue]) -> JsonValue:
    """Return the benchmark ground-truth payload."""

    eval_data: JsonObject = require_object(
        instance.get("eval_data"),
        "instance.eval_data",
    )
    return eval_data.get("answer")


def evaluate_success(
    evaluator: EvaluateOne,
    raw_output: str,
    instance: JsonObject,
) -> JsonObject:
    """Run the native evaluator and validate its correctness flag."""

    raw_result: dict[str, object] = evaluator(
        raw_output,
        cast(dict[str, object], instance),
    )
    result: JsonObject = cast(JsonObject, raw_result)
    if not isinstance(result.get("correct"), bool):
        raise DownstreamDataError(
            "Native evaluator did not return a Boolean correctness flag: "
            f"result={result!r}"
        )
    return result


def evaluation_row(
    instance_id: str,
    instance: JsonObject,
    answer: JsonObject,
    audit: JsonObject,
    validation_ids: set[str],
    evaluator: EvaluateOne,
    result_tag: str,
    served_model: str,
    domain: str,
    arm: str,
) -> JsonObject:
    """Build one auditable per-instance K=2 evaluation row."""

    audit_status: str = require_audit_status(audit, instance_id)
    category: FailureCategory = answer_failure_category(
        answer,
        audit_status,
        instance_id,
    )
    answer_domain: str = require_string(
        answer.get("dataset"),
        f"answers:{instance_id}.dataset",
    )
    if answer_domain != domain:
        raise DownstreamDataError(
            "Answer dataset mismatch: "
            f"instance_id={instance_id}, expected={domain}, "
            f"actual={answer_domain}"
        )
    answer_arm: str = require_string(
        answer.get("method"),
        f"answers:{instance_id}.method",
    )
    if answer_arm != arm:
        raise DownstreamDataError(
            "Answer arm mismatch: "
            f"instance_id={instance_id}, expected={arm}, actual={answer_arm}"
        )
    raw_output: str = require_text(
        answer.get("raw_output"),
        f"answers:{instance_id}.raw_output",
    )
    expected_skill_ids: list[str] = normalize_string_list(
        audit.get("expected_skill_ids"),
        f"audit:{instance_id}.expected_skill_ids",
    )
    skill_ids_used: list[str] = normalize_string_list(
        answer.get("skill_ids_used"),
        f"answers:{instance_id}.skill_ids_used",
    )
    request_hash: str = require_string(
        audit.get("new_request_hash"),
        f"audit:{instance_id}.new_request_hash",
    )
    if audit_status != "reused_same_arm":
        answer_request_hash: str = require_string(
            answer.get("request_hash"),
            f"answers:{instance_id}.request_hash",
        )
        if answer_request_hash != request_hash:
            raise DownstreamDataError(
                "Answer request hash disagrees with reuse audit: "
                f"instance_id={instance_id}, answer={answer_request_hash}, "
                f"audit={request_hash}"
            )
    evaluator_result: JsonObject
    if category == "success":
        if not raw_output.strip():
            raise DownstreamDataError(
                f"Successful answer is empty: instance_id={instance_id}"
            )
        evaluator_result = evaluate_success(
            evaluator,
            raw_output,
            instance,
        )
    else:
        evaluator_result = {
            "correct": False,
            "extracted_answer": "",
            "evaluation_status": "method_failure",
        }
    return {
        "schema_version": "k2-answer-evaluation-row-v1",
        "instance_id": instance_id,
        "model": result_tag,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "correct": evaluator_result["correct"],
        "failure_category": category,
        "request_hash": request_hash,
        "expected_skill_ids": expected_skill_ids,
        "skill_ids_used": skill_ids_used,
        "is_validation": instance_id in validation_ids,
        "ground_truth": require_ground_truth(instance),
        "raw_output_sha256": sha256_text(raw_output),
        "evaluator": evaluator_result,
    }


def metric_summary(rows: Sequence[JsonObject]) -> JsonObject:
    """Compute accuracy and failure counts for one explicit support set."""

    total: int = len(rows)
    if total <= 0:
        raise DownstreamDataError("Cannot summarize an empty evaluation support")
    correct: int = sum(row.get("correct") is True for row in rows)
    failure_categories: dict[str, int] = {}
    for row in rows:
        category: str = require_string(
            row.get("failure_category"),
            "evaluation.failure_category",
        )
        failure_categories[category] = failure_categories.get(category, 0) + 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "failure_categories": failure_categories,
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
    """Evaluate one complete answer file and emit full and held-out metrics."""

    args = parse_args()
    answers_path: Path = cast(Path, args.answers).resolve()
    audit_path: Path = cast(Path, args.audit).resolve()
    instances_path: Path = cast(Path, args.instances).resolve()
    validation_source_path: Path = cast(
        Path,
        args.validation_source,
    ).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    result_tag: str = str(args.result_tag)
    served_model: str = str(args.served_model)
    domain: str = str(args.domain)
    arm: str = str(args.arm)
    expected_count: int = int(args.expected_count)
    if expected_count <= 0:
        raise ValueError(
            f"expected-count must be positive: value={expected_count}"
        )

    instances: dict[str, JsonObject] = load_instances(instances_path, domain)
    answers: dict[str, JsonObject] = index_rows(
        load_jsonl(answers_path, "answers"),
        "answers",
    )
    audits: dict[str, JsonObject] = index_rows(
        load_jsonl(audit_path, "audit"),
        "audit",
    )
    expected_ids: set[str] = set(instances)
    for context, observed_ids in (
        ("answers", set(answers)),
        ("audit", set(audits)),
    ):
        missing: list[str] = sorted(expected_ids - observed_ids)
        unexpected: list[str] = sorted(observed_ids - expected_ids)
        if missing or unexpected:
            raise DownstreamDataError(
                f"{context} coverage mismatch: missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )
    if len(instances) != expected_count:
        raise DownstreamDataError(
            "Evaluation denominator mismatch: "
            f"expected={expected_count}, actual={len(instances)}"
        )
    validation_ids: set[str] = load_validation_ids(validation_source_path)
    unexpected_validation_ids: list[str] = sorted(
        validation_ids - expected_ids
    )
    if unexpected_validation_ids:
        raise DownstreamDataError(
            "Validation IDs are outside the instance support: "
            f"sample={unexpected_validation_ids[:20]}"
        )
    evaluator: EvaluateOne = load_native_evaluator()
    rows: list[JsonObject] = [
        evaluation_row(
            instance_id,
            instances[instance_id],
            answers[instance_id],
            audits[instance_id],
            validation_ids,
            evaluator,
            result_tag,
            served_model,
            domain,
            arm,
        )
        for instance_id in instances
    ]
    heldout_rows: list[JsonObject] = [
        row for row in rows if row.get("is_validation") is False
    ]
    payload: JsonObject = {
        "schema_version": "k2-answer-evaluation-v1",
        "model": result_tag,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "metrics": {
            "full": metric_summary(rows),
            "heldout": metric_summary(heldout_rows),
        },
        "provenance": {
            "answers_sha256": sha256_file(answers_path),
            "audit_sha256": sha256_file(audit_path),
            "instances_sha256": sha256_file(instances_path),
            "validation_source_sha256": sha256_file(validation_source_path),
        },
        "details": rows,
    }
    write_json_atomic(output_path, payload)
    print(
        canonical_json(
            {
                "event": "k2_answer_evaluation_complete",
                "model": result_tag,
                "domain": domain,
                "arm": arm,
                "expected": expected_count,
                "full": payload["metrics"]["full"],
                "heldout": payload["metrics"]["heldout"],
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            }
        )
    )


if __name__ == "__main__":
    main()
