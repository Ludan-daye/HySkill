#!/usr/bin/env python3
"""Validate the complete fresh runtime-matched baseline result fleet."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

from hyskill.runtime_matched_execution import canonical_json, sha256_file
from hyskill.runtime_matched_public import (
    JsonObject,
    JsonValue,
    RuntimeMatchedPublicError,
    public_answer_row,
    public_decision_row,
    public_runtime_job,
    public_usage_job,
)


JsonScalar: TypeAlias = None | bool | int | float | str

DOMAINS: tuple[tuple[str, int, int], ...] = (
    ("theoremqa", 747, 598),
    ("logicbench", 760, 608),
    ("medcalcbench", 1100, 880),
    ("champ", 223, 179),
)
NATIVE_ARMS: tuple[str, ...] = ("always_rerank", "select_bm25")
EXPECTED_ANSWER_ROWS: int = 48_110
EXPECTED_DECISION_ROWS: int = 28_300
EXPECTED_ANSWER_JOBS: int = 68
EXPECTED_DECISION_JOBS: int = 40
EXPECTED_RUNTIME_JOBS: int = 108
EXPECTED_USAGE_JOBS: int = 108
EXPECTED_LONG_METRICS: int = 178
VALIDATION_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-fleet-validation-v1"
)
MARKER_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-formal-complete-v1"
)
BARE_AUDIT_SCHEMA_VERSION: str = (
    "runtime-matched-bare-completeness-v1"
)
NATIVE_AUDIT_SCHEMA_VERSION: str = (
    "runtime-matched-native-domain-audit-v1"
)
USAGE_SUMMARY_SCHEMA_VERSION: str = "runtime-matched-usage-summary-v1"
LONG_METRIC_SCHEMA_VERSION: str = "runtime-matched-baseline-metric-v1"
SUMMARY_SCHEMA_VERSION: str = "runtime-matched-baseline-summary-v1"
COMPARISONS_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-paired-comparisons-v1"
)


class BaselineFleetValidationError(ValueError):
    """Raised when the fresh baseline fleet violates its frozen contract."""


def parse_args() -> argparse.Namespace:
    """Parse the exact result and aggregate inventory."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--seven-models", required=True)
    parser.add_argument("--five-models", required=True)
    parser.add_argument("--usage-summary", required=True, type=Path)
    parser.add_argument("--metrics-long", required=True, type=Path)
    parser.add_argument("--metrics-summary", required=True, type=Path)
    parser.add_argument("--comparisons", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return one JSON object."""

    if not isinstance(value, dict):
        raise BaselineFleetValidationError(
            f"Expected JSON object: context={context}, value={value!r}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return one JSON list."""

    if not isinstance(value, list):
        raise BaselineFleetValidationError(
            f"Expected JSON list: context={context}, value={value!r}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return one non-empty string."""

    if not isinstance(value, str) or not value:
        raise BaselineFleetValidationError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_boolean(value: JsonValue | None, context: str) -> bool:
    """Return one Boolean."""

    if not isinstance(value, bool):
        raise BaselineFleetValidationError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


def require_nonnegative_integer(
    value: JsonValue | None,
    context: str,
) -> int:
    """Return one nonnegative integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BaselineFleetValidationError(
            f"Expected nonnegative integer: context={context}, value={value!r}"
        )
    return value


def parse_model_list(value: str, context: str) -> tuple[str, ...]:
    """Parse one ordered and duplicate-free model list."""

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


def served_model_for(result_tag: str) -> str:
    """Return the frozen endpoint model tag for one public result tag."""

    if result_tag == "qwen3.5-4b-reference":
        return "qwen3.5-4b"
    return result_tag


def load_json(path: Path, context: str) -> JsonObject:
    """Load one required JSON object."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: path={path}")
    try:
        value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise BaselineFleetValidationError(
            f"{context} is malformed: path={path}, line={error.lineno}, "
            f"column={error.colno}, message={error.msg}"
        ) from error
    return require_object(value, context)


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load one required regular or gzip JSONL file."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: path={path}")
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[JsonObject] = []
    try:
        with opener(path, "rt", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise BaselineFleetValidationError(
                        f"{context} contains a blank line: "
                        f"path={path}, line={line_number}"
                    )
                try:
                    value: JsonValue = cast(JsonValue, json.loads(line))
                except json.JSONDecodeError as error:
                    raise BaselineFleetValidationError(
                        f"{context} is malformed: path={path}, "
                        f"line={line_number}, column={error.colno}, "
                        f"message={error.msg}"
                    ) from error
                rows.append(
                    require_object(
                        value,
                        f"{context}:{path}:{line_number}",
                    )
                )
    except OSError as error:
        raise BaselineFleetValidationError(
            f"{context} cannot be read: path={path}, error={error}"
        ) from error
    if not rows:
        raise BaselineFleetValidationError(
            f"{context} contains no rows: path={path}"
        )
    return rows


def require_identity(
    row: Mapping[str, JsonValue],
    expected: Mapping[str, JsonValue],
    context: str,
) -> None:
    """Require exact identity fields."""

    mismatches: list[str] = [
        f"{field}:expected={value!r},actual={row.get(field)!r}"
        for field, value in expected.items()
        if row.get(field) != value
    ]
    if mismatches:
        raise BaselineFleetValidationError(
            f"Identity mismatch: context={context}, mismatches={mismatches}"
        )


def verify_evidence_file(
    model_root: Path,
    directory: str,
    evidence: Mapping[str, JsonValue],
    context: str,
) -> Path:
    """Resolve one synced artifact by basename and verify its SHA."""

    remote_path: str = require_string(
        evidence.get("path"),
        f"{context}.path",
    )
    expected_sha: str = require_string(
        evidence.get("sha256"),
        f"{context}.sha256",
    )
    path: Path = model_root / directory / Path(remote_path).name
    if not path.is_file():
        raise FileNotFoundError(
            f"Synced artifact is missing: context={context}, path={path}"
        )
    actual_sha: str = sha256_file(path)
    if actual_sha != expected_sha:
        raise BaselineFleetValidationError(
            f"Synced artifact SHA mismatch: context={context}, "
            f"expected={expected_sha}, actual={actual_sha}, path={path}"
        )
    return path


def generation_contract(stage: str, arm: str) -> JsonObject:
    """Return the frozen generation fields for one job."""

    if stage == "answer":
        return {
            "temperature": 0.7,
            "max_tokens": 2048,
            "thinking": False,
        }
    if stage != "decision":
        raise ValueError(f"Unsupported job stage: stage={stage}")
    if arm == "always_rerank":
        return {
            "temperature": 0.0,
            "max_tokens": 1024,
            "thinking": False,
            "max_parse_attempts": 3,
            "omitted_candidate_append": True,
        }
    if arm == "select_bm25":
        return {
            "temperature": 0.0,
            "max_tokens": 64,
            "thinking": False,
            "max_parse_attempts": 3,
            "rank1_fallback": True,
        }
    raise ValueError(f"Unsupported decision arm: arm={arm}")


def validate_runtime_manifest(
    path: Path,
    model: str,
    domain: str,
    arm: str,
    stage: str,
) -> tuple[JsonObject, str]:
    """Validate one path-free runtime identity and generation contract."""

    private_manifest: JsonObject = load_json(path, "runtime manifest")
    manifest_sha: str = sha256_file(path)
    served_model: str = served_model_for(model)
    try:
        public: JsonObject = public_runtime_job(
            private_manifest,
            manifest_sha,
        )
    except RuntimeMatchedPublicError as error:
        raise BaselineFleetValidationError(
            f"Runtime manifest is not publishable: path={path}, error={error}"
        ) from error
    job: JsonObject = require_object(public.get("job"), "runtime.job")
    require_identity(
        job,
        {
            "result_tag": model,
            "model": served_model,
            "domain": domain,
            "arm": arm,
        },
        f"runtime-job:{path}",
    )
    endpoint: JsonObject = require_object(
        public.get("endpoint"),
        f"runtime-job:{path}.endpoint",
    )
    require_identity(
        endpoint,
        {
            "served_model": served_model,
            "dtype": "bfloat16",
            "quantization": "none",
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
        },
        f"runtime-job:{path}.endpoint",
    )
    readback: list[JsonValue] = require_list(
        endpoint.get("models_readback"),
        f"runtime-job:{path}.endpoint.models_readback",
    )
    if served_model not in readback:
        raise BaselineFleetValidationError(
            f"Served model is absent from endpoint readback: "
            f"model={served_model}, readback={readback}, path={path}"
        )
    generation: JsonObject = require_object(
        public.get("generation"),
        f"runtime-job:{path}.generation",
    )
    require_identity(
        generation,
        generation_contract(stage, arm),
        f"runtime-job:{path}.generation",
    )
    return public, manifest_sha


def runtime_identity_key(public: Mapping[str, JsonValue]) -> str:
    """Return one model-level runtime identity excluding job/generation."""

    payload: JsonObject = {
        field: cast(JsonValue, public.get(field))
        for field in (
            "checkpoint",
            "tokenizer",
            "endpoint",
            "software",
            "hardware",
            "source",
        )
    }
    return canonical_json(payload)


def validate_evaluation(
    path: Path,
    model: str,
    domain: str,
    arm: str,
    expected_rows: int,
    expected_heldout: int,
    runtime_manifest_sha: str,
) -> list[JsonObject]:
    """Validate one evaluation and return path-free answer rows."""

    payload: JsonObject = load_json(path, "baseline evaluation")
    served_model: str = served_model_for(model)
    require_identity(
        payload,
        {
            "schema_version": "runtime-matched-baseline-evaluation-v1",
            "model": model,
            "served_model": served_model,
            "domain": domain,
            "arm": arm,
        },
        f"evaluation:{path}",
    )
    details: list[JsonObject] = [
        require_object(item, f"evaluation:{path}.details[{index}]")
        for index, item in enumerate(
            require_list(payload.get("details"), f"evaluation:{path}.details")
        )
    ]
    if len(details) != expected_rows:
        raise BaselineFleetValidationError(
            f"Evaluation row count mismatch: path={path}, "
            f"expected={expected_rows}, actual={len(details)}"
        )
    ids: list[str] = [
        require_string(row.get("instance_id"), f"evaluation:{path}.instance_id")
        for row in details
    ]
    if len(ids) != len(set(ids)):
        raise BaselineFleetValidationError(
            f"Evaluation contains duplicate instance IDs: path={path}"
        )
    heldout: int = sum(
        not require_boolean(
            row.get("is_validation"),
            f"evaluation:{path}.is_validation",
        )
        for row in details
    )
    if heldout != expected_heldout:
        raise BaselineFleetValidationError(
            f"Evaluation held-out count mismatch: path={path}, "
            f"expected={expected_heldout}, actual={heldout}"
        )
    public_rows: list[JsonObject] = []
    for row in details:
        require_identity(
            row,
            {"served_model": served_model_for(model)},
            f"evaluation-row:{path}:{row.get('instance_id')}",
        )
        if row.get("runtime_manifest_sha256") != runtime_manifest_sha:
            raise BaselineFleetValidationError(
                "Evaluation row runtime manifest mismatch: "
                f"path={path}, instance_id={row.get('instance_id')}, "
                f"expected={runtime_manifest_sha}, "
                f"actual={row.get('runtime_manifest_sha256')}"
            )
        try:
            public_rows.append(public_answer_row(row, model, domain, arm))
        except RuntimeMatchedPublicError as error:
            raise BaselineFleetValidationError(
                f"Evaluation row is not publishable: path={path}, error={error}"
            ) from error
    metrics: JsonObject = require_object(
        payload.get("metrics"),
        f"evaluation:{path}.metrics",
    )
    full: JsonObject = require_object(
        metrics.get("full"),
        f"evaluation:{path}.metrics.full",
    )
    heldout_metrics: JsonObject = require_object(
        metrics.get("heldout"),
        f"evaluation:{path}.metrics.heldout",
    )
    if (
        full.get("total") != expected_rows
        or heldout_metrics.get("total") != expected_heldout
    ):
        raise BaselineFleetValidationError(
            f"Evaluation metric denominator mismatch: path={path}"
        )
    return public_rows


def validate_decisions(
    path: Path,
    model: str,
    domain: str,
    arm: str,
    expected_rows: int,
    runtime_manifest_sha: str,
) -> list[JsonObject]:
    """Validate one native decision file and return public rows."""

    rows: list[JsonObject] = load_jsonl(path, "native decisions")
    if len(rows) != expected_rows:
        raise BaselineFleetValidationError(
            f"Decision row count mismatch: path={path}, "
            f"expected={expected_rows}, actual={len(rows)}"
        )
    ids: list[str] = [
        require_string(row.get("instance_id"), f"decisions:{path}.instance_id")
        for row in rows
    ]
    if len(ids) != len(set(ids)):
        raise BaselineFleetValidationError(
            f"Decision file contains duplicate instance IDs: path={path}"
        )
    public_rows: list[JsonObject] = []
    for row in rows:
        require_identity(
            row,
            {"served_model": served_model_for(model)},
            f"decision-row:{path}:{row.get('instance_id')}",
        )
        if row.get("runtime_manifest_sha256") != runtime_manifest_sha:
            raise BaselineFleetValidationError(
                "Decision runtime manifest mismatch: "
                f"path={path}, instance_id={row.get('instance_id')}, "
                f"expected={runtime_manifest_sha}, "
                f"actual={row.get('runtime_manifest_sha256')}"
            )
        try:
            public_rows.append(public_decision_row(row, model, domain, arm))
        except RuntimeMatchedPublicError as error:
            raise BaselineFleetValidationError(
                f"Decision row is not publishable: path={path}, error={error}"
            ) from error
    return public_rows


def validate_bare_model(
    model_root: Path,
    model: str,
) -> tuple[list[JsonObject], list[JsonObject], list[JsonObject]]:
    """Validate all four Bare jobs for one model."""

    audit_path: Path = model_root / "audits" / "bare-completeness.json"
    audit: JsonObject = load_json(audit_path, "Bare completeness audit")
    served_model: str = served_model_for(model)
    require_identity(
        audit,
        {
            "schema_version": BARE_AUDIT_SCHEMA_VERSION,
            "model": model,
            "served_model": served_model,
            "arm": "bare",
            "expected_rows": 2830,
            "observed_rows": 2830,
            "fresh_only": True,
            "reused_same_arm": 0,
            "unresolved": 0,
            "valid": True,
        },
        f"bare-audit:{model}",
    )
    domains: JsonObject = require_object(
        audit.get("domains"),
        f"bare-audit:{model}.domains",
    )
    answer_rows: list[JsonObject] = []
    runtime_rows: list[JsonObject] = []
    source_files: list[JsonObject] = [
        {
            "path": str(audit_path.relative_to(model_root)),
            "sha256": sha256_file(audit_path),
        }
    ]
    for domain, expected_rows, expected_heldout in DOMAINS:
        domain_audit: JsonObject = require_object(
            domains.get(domain),
            f"bare-audit:{model}.domains.{domain}",
        )
        require_identity(
            domain_audit,
            {
                "expected_rows": expected_rows,
                "observed_rows": expected_rows,
                "valid": True,
            },
            f"bare-audit:{model}:{domain}",
        )
        answers_path: Path = verify_evidence_file(
            model_root,
            "answers",
            require_object(
                domain_audit.get("answers"),
                f"bare-audit:{model}:{domain}.answers",
            ),
            f"bare-audit:{model}:{domain}.answers",
        )
        evaluation_evidence: JsonObject = require_object(
            domain_audit.get("evaluation"),
            f"bare-audit:{model}:{domain}.evaluation",
        )
        evaluation_path: Path = verify_evidence_file(
            model_root,
            "eval",
            evaluation_evidence,
            f"bare-audit:{model}:{domain}.evaluation",
        )
        manifest_path: Path = (
            model_root / "runtime" / f"{domain}-bare.manifest.json"
        )
        runtime_public, manifest_sha = validate_runtime_manifest(
            manifest_path,
            model,
            domain,
            "bare",
            "answer",
        )
        answer_rows.extend(
            validate_evaluation(
                evaluation_path,
                model,
                domain,
                "bare",
                expected_rows,
                expected_heldout,
                manifest_sha,
            )
        )
        runtime_rows.append(runtime_public)
        source_files.extend(
            (
                {
                    "path": str(answers_path.relative_to(model_root)),
                    "sha256": sha256_file(answers_path),
                },
                {
                    "path": str(evaluation_path.relative_to(model_root)),
                    "sha256": sha256_file(evaluation_path),
                },
                {
                    "path": str(manifest_path.relative_to(model_root)),
                    "sha256": manifest_sha,
                },
            )
        )
    return answer_rows, runtime_rows, source_files


def native_audit_index(
    model_root: Path,
    model: str,
) -> dict[tuple[str, str], tuple[Path, JsonObject]]:
    """Index exactly eight completed native-domain audits."""

    output: dict[tuple[str, str], tuple[Path, JsonObject]] = {}
    served_model: str = served_model_for(model)
    for path in sorted((model_root / "audits").glob("*.audit.json")):
        audit: JsonObject = load_json(path, "native-domain audit")
        if audit.get("schema_version") != NATIVE_AUDIT_SCHEMA_VERSION:
            continue
        require_identity(
            audit,
            {
                "model": model,
                "served_model": served_model,
                "fresh_only": True,
                "reused_same_arm": 0,
                "unresolved": 0,
                "valid": True,
            },
            f"native-audit:{path}",
        )
        domain: str = require_string(
            audit.get("domain"),
            f"native-audit:{path}.domain",
        )
        arm: str = require_string(
            audit.get("arm"),
            f"native-audit:{path}.arm",
        )
        key: tuple[str, str] = (domain, arm)
        if key in output:
            raise BaselineFleetValidationError(
                f"Duplicate native-domain audit: model={model}, key={key}"
            )
        output[key] = (path, audit)
    expected_keys: set[tuple[str, str]] = {
        (domain, arm)
        for domain, _count, _heldout in DOMAINS
        for arm in NATIVE_ARMS
    }
    if set(output) != expected_keys:
        raise BaselineFleetValidationError(
            f"Native audit inventory mismatch: model={model}, "
            f"missing={sorted(expected_keys - set(output))}, "
            f"unexpected={sorted(set(output) - expected_keys)}"
        )
    return output


def validate_native_model(
    model_root: Path,
    model: str,
) -> tuple[
    list[JsonObject],
    list[JsonObject],
    list[JsonObject],
    list[JsonObject],
]:
    """Validate all Rerank and Select jobs for one eligible model."""

    audits: dict[tuple[str, str], tuple[Path, JsonObject]] = (
        native_audit_index(model_root, model)
    )
    answer_rows: list[JsonObject] = []
    decision_rows: list[JsonObject] = []
    runtime_rows: list[JsonObject] = []
    source_files: list[JsonObject] = []
    for domain, expected_rows, expected_heldout in DOMAINS:
        for arm in NATIVE_ARMS:
            audit_path, audit = audits[(domain, arm)]
            require_identity(
                audit,
                {
                    "domain": domain,
                    "arm": arm,
                    "expected_rows": expected_rows,
                    "observed_decisions": expected_rows,
                    "observed_answers": expected_rows,
                },
                f"native-audit:{model}:{domain}:{arm}",
            )
            artifacts: JsonObject = require_object(
                audit.get("artifacts"),
                f"native-audit:{model}:{domain}:{arm}.artifacts",
            )
            decisions_path: Path = verify_evidence_file(
                model_root,
                "decisions",
                require_object(
                    artifacts.get("decisions"),
                    "native-audit.artifacts.decisions",
                ),
                f"native-audit:{model}:{domain}:{arm}.decisions",
            )
            answers_path: Path = verify_evidence_file(
                model_root,
                "answers",
                require_object(
                    artifacts.get("answers"),
                    "native-audit.artifacts.answers",
                ),
                f"native-audit:{model}:{domain}:{arm}.answers",
            )
            evaluation_path: Path = verify_evidence_file(
                model_root,
                "eval",
                require_object(
                    artifacts.get("evaluation"),
                    "native-audit.artifacts.evaluation",
                ),
                f"native-audit:{model}:{domain}:{arm}.evaluation",
            )
            decision_manifest_path: Path = verify_evidence_file(
                model_root,
                "runtime",
                require_object(
                    artifacts.get("decision_manifest"),
                    "native-audit.artifacts.decision_manifest",
                ),
                f"native-audit:{model}:{domain}:{arm}.decision_manifest",
            )
            answer_manifest_path: Path = verify_evidence_file(
                model_root,
                "runtime",
                require_object(
                    artifacts.get("answer_manifest"),
                    "native-audit.artifacts.answer_manifest",
                ),
                f"native-audit:{model}:{domain}:{arm}.answer_manifest",
            )
            decision_runtime, decision_manifest_sha = (
                validate_runtime_manifest(
                    decision_manifest_path,
                    model,
                    domain,
                    arm,
                    "decision",
                )
            )
            answer_runtime, answer_manifest_sha = validate_runtime_manifest(
                answer_manifest_path,
                model,
                domain,
                arm,
                "answer",
            )
            decision_rows.extend(
                validate_decisions(
                    decisions_path,
                    model,
                    domain,
                    arm,
                    expected_rows,
                    decision_manifest_sha,
                )
            )
            answer_rows.extend(
                validate_evaluation(
                    evaluation_path,
                    model,
                    domain,
                    arm,
                    expected_rows,
                    expected_heldout,
                    answer_manifest_sha,
                )
            )
            runtime_rows.extend((decision_runtime, answer_runtime))
            source_files.append(
                {
                    "path": str(audit_path.relative_to(model_root)),
                    "sha256": sha256_file(audit_path),
                }
            )
            for source_path in (
                decisions_path,
                answers_path,
                evaluation_path,
                decision_manifest_path,
                answer_manifest_path,
            ):
                source_files.append(
                    {
                        "path": str(source_path.relative_to(model_root)),
                        "sha256": sha256_file(source_path),
                    }
                )
    return answer_rows, decision_rows, runtime_rows, source_files


def reject_unsupported_native_outputs(model_root: Path, model: str) -> None:
    """Reject fabricated native outputs for a Bare-only model."""

    unexpected: list[str] = []
    for directory in ("answers", "decisions", "eval", "runtime", "audits"):
        directory_path: Path = model_root / directory
        if not directory_path.is_dir():
            continue
        for path in directory_path.iterdir():
            if any(arm in path.name for arm in NATIVE_ARMS):
                unexpected.append(str(path.relative_to(model_root)))
            if "select-bm25" in path.name or "always-rerank" in path.name:
                unexpected.append(str(path.relative_to(model_root)))
    if unexpected:
        raise BaselineFleetValidationError(
            f"Bare-only model contains unsupported native outputs: "
            f"model={model}, files={sorted(set(unexpected))[:40]}"
        )


def validate_model(
    result_root: Path,
    model: str,
    native_eligible: bool,
) -> JsonObject:
    """Validate one complete model result tree."""

    model_root: Path = result_root / model
    if not model_root.is_dir():
        raise FileNotFoundError(
            f"Model result directory does not exist: model={model}, "
            f"path={model_root}"
        )
    answer_rows, runtime_rows, source_files = validate_bare_model(
        model_root,
        model,
    )
    decision_rows: list[JsonObject] = []
    if native_eligible:
        (
            native_answers,
            native_decisions,
            native_runtime,
            native_sources,
        ) = validate_native_model(model_root, model)
        answer_rows.extend(native_answers)
        decision_rows.extend(native_decisions)
        runtime_rows.extend(native_runtime)
        source_files.extend(native_sources)
    else:
        reject_unsupported_native_outputs(model_root, model)
    expected_answers: int = 8490 if native_eligible else 2830
    expected_decisions: int = 5660 if native_eligible else 0
    expected_runtime: int = 20 if native_eligible else 4
    if (
        len(answer_rows) != expected_answers
        or len(decision_rows) != expected_decisions
        or len(runtime_rows) != expected_runtime
    ):
        raise BaselineFleetValidationError(
            f"Model aggregate count mismatch: model={model}, "
            f"answers={len(answer_rows)}/{expected_answers}, "
            f"decisions={len(decision_rows)}/{expected_decisions}, "
            f"runtime={len(runtime_rows)}/{expected_runtime}"
        )
    runtime_keys: set[str] = {
        runtime_identity_key(row) for row in runtime_rows
    }
    if len(runtime_keys) != 1:
        raise BaselineFleetValidationError(
            f"Model jobs do not share one runtime identity: "
            f"model={model}, identities={len(runtime_keys)}"
        )
    unique_sources: dict[str, str] = {}
    for source in source_files:
        relative_path: str = require_string(
            source.get("path"),
            f"model:{model}.source.path",
        )
        digest: str = require_string(
            source.get("sha256"),
            f"model:{model}.source.sha256",
        )
        previous: str | None = unique_sources.get(relative_path)
        if previous is not None and previous != digest:
            raise BaselineFleetValidationError(
                f"Source path has conflicting SHA values: "
                f"model={model}, path={relative_path}"
            )
        unique_sources[relative_path] = digest
    marker: JsonObject = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "model": model,
        "native_eligible": native_eligible,
        "answer_rows": len(answer_rows),
        "decision_rows": len(decision_rows),
        "answer_jobs": 12 if native_eligible else 4,
        "decision_jobs": 8 if native_eligible else 0,
        "runtime_jobs": len(runtime_rows),
        "fresh_only": True,
        "reused_same_arm": 0,
        "runtime_identity_count": 1,
        "source_files": [
            {"path": path, "sha256": unique_sources[path]}
            for path in sorted(unique_sources)
        ],
        "valid": True,
    }
    marker_path: Path = model_root / "FORMAL_COMPLETE"
    write_immutable_json(marker_path, marker)
    return {
        "model": model,
        "native_eligible": native_eligible,
        "answer_rows": len(answer_rows),
        "decision_rows": len(decision_rows),
        "answer_jobs": 12 if native_eligible else 4,
        "decision_jobs": 8 if native_eligible else 0,
        "runtime_jobs": len(runtime_rows),
        "runtime_identity_sha256": sha256_text(next(iter(runtime_keys))),
        "marker_sha256": sha256_file(marker_path),
        "marker_path": str(marker_path),
    }


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of one UTF-8 string."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def expected_usage_keys(
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> set[tuple[str, str, str, str]]:
    """Return the frozen 108 usage-job identities."""

    eligible: set[str] = set(five_models)
    keys: set[tuple[str, str, str, str]] = set()
    for model in seven_models:
        for domain, _count, _heldout in DOMAINS:
            keys.add((model, domain, "bare", "answer"))
            if model not in eligible:
                continue
            for arm in NATIVE_ARMS:
                keys.add((model, domain, arm, "decision"))
                keys.add((model, domain, arm, "answer"))
    return keys


def validate_usage_summary(
    path: Path,
    result_root: Path,
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> JsonObject:
    """Validate the exact path-bound usage inventory and public projection."""

    payload: JsonObject = load_json(path, "usage summary")
    require_identity(
        payload,
        {"schema_version": USAGE_SUMMARY_SCHEMA_VERSION},
        "usage-summary",
    )
    jobs: list[JsonObject] = [
        require_object(item, f"usage-summary.jobs[{index}]")
        for index, item in enumerate(
            require_list(payload.get("jobs"), "usage-summary.jobs")
        )
    ]
    if len(jobs) != EXPECTED_USAGE_JOBS:
        raise BaselineFleetValidationError(
            f"Usage job count mismatch: expected={EXPECTED_USAGE_JOBS}, "
            f"actual={len(jobs)}"
        )
    expected_keys: set[tuple[str, str, str, str]] = expected_usage_keys(
        seven_models,
        five_models,
    )
    observed_keys: set[tuple[str, str, str, str]] = set()
    aggregate_fields: tuple[str, ...] = (
        "http_calls",
        "response_calls",
        "error_calls",
        "usage_reported_calls",
        "usage_missing_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )
    totals: dict[str, int] = {field: 0 for field in aggregate_fields}
    for job in jobs:
        model: str = require_string(job.get("model"), "usage-job.model")
        domain: str = require_string(job.get("domain"), "usage-job.domain")
        arm: str = require_string(job.get("arm"), "usage-job.arm")
        stage: str = require_string(job.get("stage"), "usage-job.stage")
        key: tuple[str, str, str, str] = (model, domain, arm, stage)
        if key in observed_keys:
            raise BaselineFleetValidationError(
                f"Duplicate usage job identity: key={key}"
            )
        observed_keys.add(key)
        public_usage_job(job)
        remote_path: str = require_string(job.get("path"), "usage-job.path")
        local_path: Path = result_root / model / "logs" / Path(remote_path).name
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Usage source log is missing: key={key}, path={local_path}"
            )
        expected_sha: str = require_string(
            job.get("sha256"),
            "usage-job.sha256",
        )
        actual_sha: str = sha256_file(local_path)
        if actual_sha != expected_sha:
            raise BaselineFleetValidationError(
                f"Usage source SHA mismatch: key={key}, "
                f"expected={expected_sha}, actual={actual_sha}"
            )
        for field in aggregate_fields:
            totals[field] += require_nonnegative_integer(
                job.get(field),
                f"usage-job.{field}",
            )
    if observed_keys != expected_keys:
        raise BaselineFleetValidationError(
            f"Usage job support mismatch: "
            f"missing={sorted(expected_keys - observed_keys)[:40]}, "
            f"unexpected={sorted(observed_keys - expected_keys)[:40]}"
        )
    overall: JsonObject = require_object(
        payload.get("overall"),
        "usage-summary.overall",
    )
    for field, expected in totals.items():
        if overall.get(field) != expected:
            raise BaselineFleetValidationError(
                f"Usage overall mismatch: field={field}, "
                f"expected={expected}, actual={overall.get(field)!r}"
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "job_logs": len(jobs),
        "http_calls": totals["http_calls"],
        "usage_reported_calls": totals["usage_reported_calls"],
        "usage_missing_calls": totals["usage_missing_calls"],
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["total_tokens"],
    }


def validate_aggregate_products(
    metrics_long_path: Path,
    metrics_summary_path: Path,
    comparisons_path: Path,
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> JsonObject:
    """Validate the three fresh-only statistical products."""

    long_rows: list[JsonObject] = load_jsonl(
        metrics_long_path,
        "baseline long metrics",
    )
    if len(long_rows) != EXPECTED_LONG_METRICS:
        raise BaselineFleetValidationError(
            f"Long metric count mismatch: expected={EXPECTED_LONG_METRICS}, "
            f"actual={len(long_rows)}"
        )
    for index, row in enumerate(long_rows):
        require_identity(
            row,
            {"schema_version": LONG_METRIC_SCHEMA_VERSION},
            f"long-metric:{index}",
        )
    summary: JsonObject = load_json(
        metrics_summary_path,
        "baseline metrics summary",
    )
    require_identity(
        summary,
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "seven_models": list(seven_models),
            "five_models": list(five_models),
            "expected_total_per_model": 2830,
            "expected_heldout_per_model": 2265,
            "fresh_baseline_rows": EXPECTED_ANSWER_ROWS,
            "legacy_compact_baseline_read": False,
        },
        "baseline-metrics-summary",
    )
    comparisons: JsonObject = load_json(
        comparisons_path,
        "baseline comparisons",
    )
    require_identity(
        comparisons,
        {
            "schema_version": COMPARISONS_SCHEMA_VERSION,
            "split": "heldout",
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 0,
            "comparison_count": 4,
        },
        "baseline-comparisons",
    )
    support: JsonObject = require_object(
        comparisons.get("support"),
        "baseline-comparisons.support",
    )
    require_identity(
        support,
        {
            "seven_models": list(seven_models),
            "five_models": list(five_models),
            "expected_total_per_model": 2830,
            "expected_heldout_per_model": 2265,
            "fresh_baseline_rows": EXPECTED_ANSWER_ROWS,
        },
        "baseline-comparisons.support",
    )
    comparison_rows: JsonObject = require_object(
        comparisons.get("comparisons"),
        "baseline-comparisons.comparisons",
    )
    if len(comparison_rows) != 4:
        raise BaselineFleetValidationError(
            f"Comparison inventory mismatch: actual={len(comparison_rows)}"
        )
    return {
        "metrics_long": {
            "path": str(metrics_long_path),
            "sha256": sha256_file(metrics_long_path),
            "rows": len(long_rows),
        },
        "metrics_summary": {
            "path": str(metrics_summary_path),
            "sha256": sha256_file(metrics_summary_path),
            "rows": 1,
        },
        "comparisons": {
            "path": str(comparisons_path),
            "sha256": sha256_file(comparisons_path),
            "rows": 4,
        },
    }


def write_immutable_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    """Write one derived JSON file, refusing a conflicting existing file."""

    encoded: str = canonical_json(dict(payload)) + "\n"
    if path.exists():
        current: str = path.read_text(encoding="utf-8")
        if current != encoded:
            raise FileExistsError(
                f"Refusing to replace conflicting derived evidence: path={path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(encoded, encoding="utf-8")
    temporary_path.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, JsonValue]) -> None:
    """Write one replaceable aggregate validation report atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    """Validate all fresh baseline rows and emit formal completion evidence."""

    args = parse_args()
    result_root: Path = cast(Path, args.result_root).resolve()
    seven_models: tuple[str, ...] = parse_model_list(
        str(args.seven_models),
        "seven-models",
    )
    five_models: tuple[str, ...] = parse_model_list(
        str(args.five_models),
        "five-models",
    )
    if len(seven_models) != 7 or len(five_models) != 5:
        raise ValueError(
            f"Frozen support requires seven and five models: "
            f"seven={len(seven_models)}, five={len(five_models)}"
        )
    if not set(five_models).issubset(set(seven_models)):
        raise ValueError(
            f"five-models must be a subset of seven-models: "
            f"five={five_models}, seven={seven_models}"
        )
    model_reports: list[JsonObject] = [
        validate_model(
            result_root,
            model,
            model in set(five_models),
        )
        for model in seven_models
    ]
    answer_rows: int = sum(
        require_nonnegative_integer(
            report.get("answer_rows"),
            "model-report.answer_rows",
        )
        for report in model_reports
    )
    decision_rows: int = sum(
        require_nonnegative_integer(
            report.get("decision_rows"),
            "model-report.decision_rows",
        )
        for report in model_reports
    )
    answer_jobs: int = sum(
        require_nonnegative_integer(
            report.get("answer_jobs"),
            "model-report.answer_jobs",
        )
        for report in model_reports
    )
    decision_jobs: int = sum(
        require_nonnegative_integer(
            report.get("decision_jobs"),
            "model-report.decision_jobs",
        )
        for report in model_reports
    )
    runtime_jobs: int = sum(
        require_nonnegative_integer(
            report.get("runtime_jobs"),
            "model-report.runtime_jobs",
        )
        for report in model_reports
    )
    expected_totals: tuple[tuple[str, int, int], ...] = (
        ("answer_rows", answer_rows, EXPECTED_ANSWER_ROWS),
        ("decision_rows", decision_rows, EXPECTED_DECISION_ROWS),
        ("answer_jobs", answer_jobs, EXPECTED_ANSWER_JOBS),
        ("decision_jobs", decision_jobs, EXPECTED_DECISION_JOBS),
        ("runtime_jobs", runtime_jobs, EXPECTED_RUNTIME_JOBS),
    )
    mismatches: list[str] = [
        f"{name}:expected={expected},actual={actual}"
        for name, actual, expected in expected_totals
        if actual != expected
    ]
    if mismatches:
        raise BaselineFleetValidationError(
            f"Fleet total mismatch: mismatches={mismatches}"
        )
    usage: JsonObject = validate_usage_summary(
        cast(Path, args.usage_summary).resolve(),
        result_root,
        seven_models,
        five_models,
    )
    aggregates: JsonObject = validate_aggregate_products(
        cast(Path, args.metrics_long).resolve(),
        cast(Path, args.metrics_summary).resolve(),
        cast(Path, args.comparisons).resolve(),
        seven_models,
        five_models,
    )
    output_path: Path = cast(Path, args.output).resolve()
    payload: JsonObject = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "support": {
            "seven_models": list(seven_models),
            "five_models": list(five_models),
            "domains": [domain for domain, _count, _heldout in DOMAINS],
            "unsupported_native_models": [
                model for model in seven_models if model not in set(five_models)
            ],
        },
        "totals": {
            "answer_rows": answer_rows,
            "decision_rows": decision_rows,
            "answer_jobs": answer_jobs,
            "decision_jobs": decision_jobs,
            "runtime_jobs": runtime_jobs,
            "usage_jobs": usage["job_logs"],
        },
        "models": model_reports,
        "usage": usage,
        "aggregates": aggregates,
        "semantics": {
            "fresh_only": True,
            "legacy_compact_baseline_read": False,
            "unsupported_native_cells": "unavailable, never zero-filled",
            "method_failures_remain_in_denominator": True,
            "token_source": "actual OpenAI-compatible response usage",
            "missing_usage_is_never_imputed": True,
        },
        "valid": True,
    }
    write_json_atomic(output_path, payload)
    print(
        canonical_json(
            {
                "event": "runtime_matched_baseline_fleet_validated",
                "answer_rows": answer_rows,
                "decision_rows": decision_rows,
                "runtime_jobs": runtime_jobs,
                "usage_jobs": usage["job_logs"],
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "valid": True,
            }
        )
    )


if __name__ == "__main__":
    main()
