#!/usr/bin/env python3
"""Evaluate one fresh runtime-matched baseline answer job."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol, TypeAlias, TypedDict, cast

from hyskill.runtime_matched_execution import (
    canonical_json,
    require_sha256,
    sha256_file,
    sha256_text,
)


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
BaselineArm: TypeAlias = Literal[
    "bare",
    "always_rerank",
    "select_bm25",
]
FailureCategory: TypeAlias = Literal[
    "success",
    "method_failure",
    "infra_transient",
    "unclassified_error",
]

ANSWER_SCHEMA_VERSION: str = "runtime-matched-baseline-answer-v1"
EVALUATION_ROW_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-evaluation-row-v1"
)
EVALUATION_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-evaluation-v1"
)
BASELINE_ARMS: frozenset[str] = frozenset(
    {
        "bare",
        "always_rerank",
        "select_bm25",
    }
)
RESOLVED_ANSWER_CATEGORIES: frozenset[str] = frozenset(
    {
        "success",
        "method_failure",
    }
)


class BaselineEvaluationError(ValueError):
    """Raised when fresh baseline evidence is incomplete or inconsistent."""


class EvaluateOne(Protocol):
    """Native SR-Agents evaluator dispatch contract."""

    def __call__(
        self,
        raw_output: str,
        instance: dict[str, object],
    ) -> dict[str, object]:
        """Evaluate one model output against one benchmark instance."""


class EvaluationRow(TypedDict):
    """One fresh per-instance correctness record."""

    schema_version: str
    instance_id: str
    model: str
    served_model: str
    domain: str
    arm: BaselineArm
    correct: bool
    failure_category: FailureCategory
    is_validation: bool
    ground_truth: JsonValue
    raw_output_sha256: str
    answer_payload_hash: str
    execution_request_hash: str
    runtime_manifest_sha256: str
    skill_ids_used: list[str]
    evaluator: JsonObject


def parse_args() -> argparse.Namespace:
    """Parse explicit inputs for one model-domain-arm evaluation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--validation-source", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=tuple(sorted(BASELINE_ARMS)),
    )
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return one JSON object with source context."""

    if not isinstance(value, dict):
        raise BaselineEvaluationError(
            "Expected JSON object: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return one JSON list with source context."""

    if not isinstance(value, list):
        raise BaselineEvaluationError(
            "Expected JSON list: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return one non-empty string with source context."""

    if not isinstance(value, str) or not value:
        raise BaselineEvaluationError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_text(value: JsonValue | None, context: str) -> str:
    """Return one string, including an explicitly empty string."""

    if not isinstance(value, str):
        raise BaselineEvaluationError(
            f"Expected string: context={context}, value={value!r}"
        )
    return value


def require_boolean(value: JsonValue | None, context: str) -> bool:
    """Return one Boolean with source context."""

    if not isinstance(value, bool):
        raise BaselineEvaluationError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


def require_string_list(
    value: JsonValue | None,
    context: str,
) -> list[str]:
    """Return one duplicate-free JSON string list."""

    raw_values: list[JsonValue] = require_list(value, context)
    values: list[str] = [
        require_string(item, f"{context}[{index}]")
        for index, item in enumerate(raw_values)
    ]
    if len(values) != len(set(values)):
        raise BaselineEvaluationError(
            f"String list contains duplicates: context={context}"
        )
    return values


def load_json(path: Path, context: str) -> JsonValue:
    """Load one UTF-8 JSON file."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    try:
        return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise BaselineEvaluationError(
            f"{context} JSON is malformed: path={path}, "
            f"line={error.lineno}, column={error.colno}, message={error.msg}"
        ) from error


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load one non-empty UTF-8 JSONL file."""

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
                raise BaselineEvaluationError(
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
        raise BaselineEvaluationError(
            f"{context} JSONL is empty: path={path}"
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
            raise BaselineEvaluationError(
                f"{context} contains duplicate instance ID: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def load_instances(path: Path, domain: str) -> dict[str, JsonObject]:
    """Load one exact-domain benchmark instance set."""

    raw_rows: list[JsonValue] = require_list(
        load_json(path, "instances"),
        "instances",
    )
    rows: list[JsonObject] = [
        require_object(row, f"instances[{index}]")
        for index, row in enumerate(raw_rows)
    ]
    output: dict[str, JsonObject] = index_rows(rows, "instances")
    for instance_id, instance in output.items():
        instance_domain: str = require_string(
            instance.get("dataset"),
            f"instances:{instance_id}.dataset",
        )
        if instance_domain != domain:
            raise BaselineEvaluationError(
                "Instance domain mismatch: "
                f"instance_id={instance_id}, expected={domain}, "
                f"actual={instance_domain}"
            )
    return output


def load_validation_ids(path: Path) -> set[str]:
    """Load frozen K=2 calibration IDs as the held-out authority."""

    payload: JsonObject = require_object(
        load_json(path, "validation-source"),
        "validation-source",
    )
    raw_ids: list[JsonValue] = require_list(
        payload.get("val_ids"),
        "validation-source.val_ids",
    )
    validation_ids: list[str] = [
        require_string(
            instance_id,
            f"validation-source.val_ids[{index}]",
        )
        for index, instance_id in enumerate(raw_ids)
    ]
    if len(validation_ids) != len(set(validation_ids)):
        raise BaselineEvaluationError(
            "Validation source contains duplicate instance IDs: "
            f"path={path}"
        )
    return set(validation_ids)


def load_native_evaluator() -> EvaluateOne:
    """Load the frozen SR-Agents evaluator dispatcher."""

    try:
        evaluate_module: ModuleType = importlib.import_module(
            "sragents.evaluate"
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SR-Agents evaluator is unavailable. Add "
            "external/SR-Agents/src to PYTHONPATH before evaluation."
        ) from error
    evaluator: object = getattr(evaluate_module, "evaluate", None)
    if not callable(evaluator):
        raise RuntimeError(
            "SR-Agents evaluator dispatcher is not callable: "
            "module=sragents.evaluate, attribute=evaluate"
        )
    return cast(EvaluateOne, evaluator)


def require_answer_category(
    answer: Mapping[str, JsonValue],
    instance_id: str,
) -> FailureCategory:
    """Return one final answer category and reject unresolved failures."""

    category_value: JsonValue | None = answer.get("failure_category")
    if (
        not isinstance(category_value, str)
        or category_value not in RESOLVED_ANSWER_CATEGORIES
    ):
        raise BaselineEvaluationError(
            "Evaluation accepts only resolved fresh answer outcomes: "
            f"instance_id={instance_id}, category={category_value!r}"
        )
    return cast(FailureCategory, category_value)


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
    require_boolean(result.get("correct"), "native-evaluator.correct")
    return result


def evaluation_row(
    instance_id: str,
    instance: JsonObject,
    answer: JsonObject,
    validation_ids: set[str],
    evaluator: EvaluateOne,
    result_tag: str,
    served_model: str,
    domain: str,
    arm: BaselineArm,
) -> EvaluationRow:
    """Build one auditable fresh baseline evaluation row."""

    if answer.get("schema_version") != ANSWER_SCHEMA_VERSION:
        raise BaselineEvaluationError(
            "Answer schema is not the fresh runtime-matched schema: "
            f"instance_id={instance_id}, "
            f"schema={answer.get('schema_version')!r}"
        )
    if answer.get("stage") != "answer":
        raise BaselineEvaluationError(
            "Answer row has an invalid execution stage: "
            f"instance_id={instance_id}, stage={answer.get('stage')!r}"
        )
    if answer.get("reused_same_arm") is not False:
        raise BaselineEvaluationError(
            "Fresh runtime-matched answers must not reuse legacy rows: "
            f"instance_id={instance_id}, "
            f"reused_same_arm={answer.get('reused_same_arm')!r}"
        )
    expected_fields: tuple[tuple[str, str], ...] = (
        ("model", result_tag),
        ("served_model", served_model),
        ("domain", domain),
        ("arm", arm),
    )
    for field_name, expected_value in expected_fields:
        actual_value: str = require_string(
            answer.get(field_name),
            f"answers:{instance_id}.{field_name}",
        )
        if actual_value != expected_value:
            raise BaselineEvaluationError(
                "Answer identity mismatch: "
                f"instance_id={instance_id}, field={field_name}, "
                f"expected={expected_value}, actual={actual_value}"
            )
    category: FailureCategory = require_answer_category(
        answer,
        instance_id,
    )
    raw_output: str = require_text(
        answer.get("raw_output"),
        f"answers:{instance_id}.raw_output",
    )
    if category == "success":
        if not raw_output.strip():
            raise BaselineEvaluationError(
                f"Successful answer is empty: instance_id={instance_id}"
            )
        evaluator_result: JsonObject = evaluate_success(
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
        "failure_category": category,
        "is_validation": instance_id in validation_ids,
        "ground_truth": require_ground_truth(instance),
        "raw_output_sha256": sha256_text(raw_output),
        "answer_payload_hash": require_sha256(
            answer.get("answer_payload_hash"),
            f"answers:{instance_id}.answer_payload_hash",
        ),
        "execution_request_hash": require_sha256(
            answer.get("execution_request_hash"),
            f"answers:{instance_id}.execution_request_hash",
        ),
        "runtime_manifest_sha256": require_sha256(
            answer.get("runtime_manifest_sha256"),
            f"answers:{instance_id}.runtime_manifest_sha256",
        ),
        "skill_ids_used": require_string_list(
            answer.get("skill_ids_used"),
            f"answers:{instance_id}.skill_ids_used",
        ),
        "evaluator": evaluator_result,
    }


def metric_summary(rows: Sequence[EvaluationRow]) -> JsonObject:
    """Compute accuracy and outcome counts for one explicit support."""

    if not rows:
        raise BaselineEvaluationError(
            "Cannot summarize an empty evaluation support"
        )
    correct: int = sum(row["correct"] for row in rows)
    categories: dict[str, int] = {}
    for row in rows:
        category: str = row["failure_category"]
        categories[category] = categories.get(category, 0) + 1
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "failure_categories": categories,
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
    """Evaluate one complete fresh answer file."""

    args = parse_args()
    answers_path: Path = cast(Path, args.answers).resolve()
    instances_path: Path = cast(Path, args.instances).resolve()
    validation_source_path: Path = cast(
        Path,
        args.validation_source,
    ).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    result_tag: str = str(args.result_tag)
    served_model: str = str(args.served_model)
    domain: str = str(args.domain)
    arm: BaselineArm = cast(BaselineArm, str(args.arm))
    expected_count: int = int(args.expected_count)
    if expected_count <= 0:
        raise ValueError(
            f"expected-count must be positive: value={expected_count}"
        )

    instances: dict[str, JsonObject] = load_instances(
        instances_path,
        domain,
    )
    answers: dict[str, JsonObject] = index_rows(
        load_jsonl(answers_path, "answers"),
        "answers",
    )
    expected_ids: set[str] = set(instances)
    observed_ids: set[str] = set(answers)
    if expected_ids != observed_ids:
        raise BaselineEvaluationError(
            "Answer coverage mismatch: "
            f"missing={sorted(expected_ids - observed_ids)[:20]}, "
            f"unexpected={sorted(observed_ids - expected_ids)[:20]}"
        )
    if len(instances) != expected_count:
        raise BaselineEvaluationError(
            "Evaluation denominator mismatch: "
            f"expected={expected_count}, actual={len(instances)}"
        )
    validation_ids: set[str] = load_validation_ids(validation_source_path)
    unexpected_validation_ids: list[str] = sorted(
        validation_ids - expected_ids
    )
    if unexpected_validation_ids:
        raise BaselineEvaluationError(
            "Validation IDs are outside the instance support: "
            f"sample={unexpected_validation_ids[:20]}"
        )
    evaluator: EvaluateOne = load_native_evaluator()
    rows: list[EvaluationRow] = [
        evaluation_row(
            instance_id,
            instances[instance_id],
            answers[instance_id],
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
            "instances": {
                "path": str(instances_path),
                "sha256": sha256_file(instances_path),
            },
            "validation_source": {
                "path": str(validation_source_path),
                "sha256": sha256_file(validation_source_path),
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
                "event": "runtime_matched_baseline_evaluation_complete",
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
