#!/usr/bin/env python3
"""Run one audited K=2 direct-answer job with strict resume semantics."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    FailureCategory,
    JsonLike,
    JsonObject,
    JsonValue,
    RuntimeManifest,
    SemanticArm,
    audit_record_coverage,
    canonical_json,
    classify_request_error,
    sha256_file,
    sha256_text,
    validate_failure_category,
)
from scripts.audit_k2_reuse import (
    AnswerRuntime,
    answer_hash,
    expected_skill_ids,
    load_answer_runtime,
    load_corpus,
    load_decisions,
    load_instances,
    load_manifest,
    loaded_skills,
    require_list,
    require_object,
    require_string,
)

if TYPE_CHECKING:
    from openai import OpenAI


ANSWER_RECORD_SCHEMA_VERSION: str = "k2-answer-record-v1"
MAX_INFRA_ATTEMPTS: int = 3
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0)
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 2048
THINKING: bool = False


class EngineResult(Protocol):
    """Fields returned by the native SR-Agents inference engine."""

    raw_output: str
    transcript: str | None
    skill_ids_used: list[str]
    meta: JsonObject


class DirectEngine(Protocol):
    """Native SR-Agents direct engine contract."""

    def run(
        self,
        instance: JsonObject,
        skills: list[JsonObject],
        client: OpenAI,
        model: str,
    ) -> EngineResult:
        """Execute the unchanged direct engine."""


class CreateClient(Protocol):
    """SR-Agents client factory contract."""

    def __call__(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> OpenAI:
        """Create one shared client."""


class ModelShortName(Protocol):
    """SR-Agents model label helper contract."""

    def __call__(self, model: str) -> str:
        """Return the persisted display label."""


class DirectEngineFactory(Protocol):
    """Native direct-engine constructor contract."""

    def __call__(
        self,
        *,
        temperature: float,
        max_tokens: int,
        thinking: bool,
    ) -> DirectEngine:
        """Create a frozen direct engine."""


class AnswerEngineRuntime(TypedDict):
    """Native answer components loaded from SR-Agents."""

    create_client: CreateClient
    create_engine: DirectEngineFactory
    model_short_name: ModelShortName
    request_error_types: tuple[type[Exception], ...]


class AuditRow(TypedDict):
    """Fields required from one reuse audit row."""

    instance_id: str
    arm: SemanticArm
    status: str
    needs_inference: bool
    expected_skill_ids: list[str]
    new_request_hash: str
    source_line_sha256: str | None


class AnswerLine(TypedDict):
    """One output JSONL row with raw-line provenance."""

    raw_line: str
    record: JsonObject


class RunOutcome(TypedDict):
    """One new direct-answer outcome."""

    record: JsonObject
    engine_attempts: int


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-domain-arm answer job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--decision-source", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--preseed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt-log", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
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
    parser.add_argument("--domain", required=True)
    parser.add_argument("--workers", required=True, type=int)
    return parser.parse_args()


def load_engine_runtime() -> AnswerEngineRuntime:
    """Load the unchanged native direct-answer implementation."""

    try:
        direct_module: ModuleType = importlib.import_module(
            "sragents.infer.engines.direct"
        )
        llm_module: ModuleType = importlib.import_module("sragents.llm")
        config_module: ModuleType = importlib.import_module("sragents.config")
        openai_module: ModuleType = importlib.import_module("openai")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SR-Agents runtime is unavailable. Install the project environment "
            "before running K=2 answers: "
            "command=.venv/bin/pip install --no-deps -e external/SR-Agents; "
            "install any missing packages separately from an approved China mirror"
        ) from error
    api_error_type: type[Exception] | None = cast(
        type[Exception] | None,
        getattr(openai_module, "APIError", None),
    )
    if api_error_type is None or not issubclass(api_error_type, Exception):
        raise RuntimeError(
            "Installed OpenAI client does not expose APIError for bounded retries"
        )
    return {
        "create_client": cast(
            CreateClient, getattr(llm_module, "create_llm_client")
        ),
        "create_engine": cast(
            DirectEngineFactory,
            getattr(direct_module, "DirectEngine"),
        ),
        "model_short_name": cast(
            ModelShortName, getattr(config_module, "model_short_name")
        ),
        "request_error_types": (api_error_type,),
    }


def load_jsonl(path: Path, context: str) -> list[AnswerLine]:
    """Load JSONL rows while preserving their exact source lines."""

    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL does not exist: path={path}")
    output: list[AnswerLine] = []
    with path.open(encoding="utf-8", newline="") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue
            try:
                raw_record: JsonValue = cast(JsonValue, json.loads(raw_line))
            except json.JSONDecodeError as error:
                raise DownstreamDataError(
                    f"{context} JSONL is malformed: path={path}, "
                    f"line={line_number}, column={error.colno}, message={error.msg}"
                ) from error
            record: JsonObject = require_object(
                raw_record,
                f"{context}:{path}:{line_number}",
            )
            require_string(
                record.get("instance_id"),
                f"{context}:{path}:{line_number}.instance_id",
            )
            output.append({"raw_line": raw_line, "record": record})
    return output


def load_audit(path: Path, arm: SemanticArm) -> dict[str, AuditRow]:
    """Load an exact-coverage reuse audit for one semantic arm."""

    lines: list[AnswerLine] = load_jsonl(path, "audit")
    output: dict[str, AuditRow] = {}
    for line in lines:
        record: JsonObject = line["record"]
        instance_id: str = require_string(
            record.get("instance_id"),
            "audit.instance_id",
        )
        if instance_id in output:
            raise DownstreamDataError(
                f"Reuse audit contains duplicate instance: instance_id={instance_id}"
            )
        if record.get("arm") != arm:
            raise DownstreamDataError(
                "Reuse audit arm mismatch: "
                f"instance_id={instance_id}, expected={arm}, actual={record.get('arm')!r}"
            )
        raw_status: JsonValue | None = record.get("status")
        if raw_status not in ("reused_same_arm", "needs_inference", "rejected"):
            raise DownstreamDataError(
                f"Reuse audit has invalid status: instance_id={instance_id}, status={raw_status!r}"
            )
        needs_inference: JsonValue | None = record.get("needs_inference")
        if not isinstance(needs_inference, bool):
            raise DownstreamDataError(
                "Reuse audit needs_inference must be Boolean: "
                f"instance_id={instance_id}, value={needs_inference!r}"
            )
        expected: list[JsonValue] = require_list(
            record.get("expected_skill_ids"),
            f"audit:{instance_id}.expected_skill_ids",
        )
        expected_skill_ids_value: list[str] = [
            require_string(
                skill_id,
                f"audit:{instance_id}.expected_skill_ids[{index}]",
            )
            for index, skill_id in enumerate(expected)
        ]
        raw_source_line_sha: JsonValue | None = record.get("source_line_sha256")
        source_line_sha: str | None = (
            raw_source_line_sha
            if isinstance(raw_source_line_sha, str)
            else None
        )
        output[instance_id] = {
            "instance_id": instance_id,
            "arm": arm,
            "status": cast(str, raw_status),
            "needs_inference": needs_inference,
            "expected_skill_ids": expected_skill_ids_value,
            "new_request_hash": require_string(
                record.get("new_request_hash"),
                f"audit:{instance_id}.new_request_hash",
            ),
            "source_line_sha256": source_line_sha,
        }
    return output


def indexed_answer_lines(
    lines: Sequence[AnswerLine],
    context: str,
) -> dict[str, AnswerLine]:
    """Index unique answer lines by instance ID."""

    output: dict[str, AnswerLine] = {}
    for line in lines:
        instance_id: str = require_string(
            line["record"].get("instance_id"),
            f"{context}.instance_id",
        )
        if instance_id in output:
            raise DownstreamDataError(
                f"{context} contains duplicate instance: instance_id={instance_id}"
            )
        output[instance_id] = line
    return output


def initialize_output(
    output_path: Path,
    preseed_path: Path,
    audit: Mapping[str, AuditRow],
    write_lock: threading.Lock,
) -> None:
    """Create or extend output with exactly the approved preseed records."""

    preseed_lines: list[AnswerLine] = load_jsonl(preseed_path, "preseed")
    preseed_index: dict[str, AnswerLine] = indexed_answer_lines(
        preseed_lines,
        "preseed",
    )
    expected_reused_ids: set[str] = {
        instance_id
        for instance_id, row in audit.items()
        if row["status"] == "reused_same_arm"
    }
    if set(preseed_index) != expected_reused_ids:
        raise DownstreamDataError(
            "Preseed IDs do not match approved reuse audit: "
            f"missing={sorted(expected_reused_ids - set(preseed_index))[:20]}, "
            f"unexpected={sorted(set(preseed_index) - expected_reused_ids)[:20]}"
        )
    for instance_id, line in preseed_index.items():
        expected_line_sha: str | None = audit[instance_id]["source_line_sha256"]
        actual_line_sha: str = sha256_text(line["raw_line"].rstrip("\r\n"))
        if expected_line_sha != actual_line_sha:
            raise DownstreamDataError(
                "Preseed source-line hash mismatch: "
                f"instance_id={instance_id}, expected={expected_line_sha}, "
                f"actual={actual_line_sha}"
            )
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path = output_path.with_name(
            f"{output_path.name}.tmp.{os.getpid()}"
        )
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            for line in preseed_lines:
                output_file.write(line["raw_line"])
                if not line["raw_line"].endswith("\n"):
                    output_file.write("\n")
        temporary_path.replace(output_path)
        return
    output_index: dict[str, AnswerLine] = indexed_answer_lines(
        load_jsonl(output_path, "answer-output"),
        "answer-output",
    )
    for instance_id, preseed_line in preseed_index.items():
        if instance_id not in output_index:
            append_raw_line(output_path, preseed_line["raw_line"], write_lock)
            continue
        if sha256_text(
            output_index[instance_id]["raw_line"].rstrip("\r\n")
        ) != sha256_text(
            preseed_line["raw_line"].rstrip("\r\n")
        ):
            raise DownstreamDataError(
                "Existing reused answer differs from approved preseed: "
                f"instance_id={instance_id}"
            )


def append_raw_line(
    path: Path,
    raw_line: str,
    write_lock: threading.Lock,
) -> None:
    """Append one complete JSONL source line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with write_lock:
        with path.open("a", encoding="utf-8", newline="") as output_file:
            output_file.write(raw_line)
            if not raw_line.endswith("\n"):
                output_file.write("\n")
            output_file.flush()


def rewrite_retryable_output(
    output_path: Path,
    audit: Mapping[str, AuditRow],
) -> dict[str, AnswerLine]:
    """Atomically drop only new unresolved infrastructure failures."""

    lines: list[AnswerLine] = load_jsonl(output_path, "answer-output")
    retained: list[AnswerLine] = []
    for line in lines:
        record: JsonObject = line["record"]
        instance_id: str = require_string(
            record.get("instance_id"),
            "answer-output.instance_id",
        )
        if audit[instance_id]["status"] == "reused_same_arm":
            retained.append(line)
            continue
        category: FailureCategory = validate_failure_category(
            record.get("failure_category")
        )
        if category != "infra_transient":
            retained.append(line)
    if len(retained) != len(lines):
        temporary_path: Path = output_path.with_name(
            f"{output_path.name}.tmp.{os.getpid()}"
        )
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            for line in retained:
                output_file.write(line["raw_line"])
                if not line["raw_line"].endswith("\n"):
                    output_file.write("\n")
        temporary_path.replace(output_path)
    return indexed_answer_lines(retained, "answer-output")


def error_context(error: Exception) -> tuple[str, int | None, str]:
    """Extract explicit exception, HTTP status, and response body."""

    exception_name: str = type(error).__name__
    raw_status_code: int | None = cast(
        int | None,
        getattr(error, "status_code", None),
    )
    status_code: int | None = (
        raw_status_code if isinstance(raw_status_code, int) else None
    )
    raw_body: JsonLike | None = cast(
        JsonLike | None,
        getattr(error, "body", None),
    )
    if raw_body is None:
        raw_response_text: str | None = cast(
            str | None,
            getattr(getattr(error, "response", None), "text", None),
        )
        raw_body = raw_response_text
    if raw_body is None:
        return exception_name, status_code, ""
    if isinstance(raw_body, str):
        return exception_name, status_code, raw_body
    try:
        return exception_name, status_code, canonical_json(raw_body)
    except DownstreamDataError:
        return exception_name, status_code, repr(raw_body)


def append_attempt(
    path: Path,
    payload: Mapping[str, JsonLike],
    write_lock: threading.Lock,
) -> None:
    """Append one structured engine attempt."""

    append_raw_line(path, canonical_json(payload), write_lock)


def failure_record(
    instance: JsonObject,
    arm: SemanticArm,
    model: str,
    model_label: str,
    expected: Sequence[str],
    request_hash: str,
    category: FailureCategory,
    error_payload: Mapping[str, JsonLike],
    engine_attempts: int,
    manifest: RuntimeManifest,
    decision_source_sha256: str,
) -> JsonObject:
    """Build one explicit failed-answer record."""

    return {
        "schema_version": ANSWER_RECORD_SCHEMA_VERSION,
        "instance_id": require_string(
            instance.get("instance_id"),
            "answer.instance_id",
        ),
        "dataset": require_string(instance.get("dataset"), "answer.dataset"),
        "method": arm,
        "model": model_label,
        "served_model": model,
        "raw_output": "",
        "skill_ids_used": [],
        "expected_skill_ids": list(expected),
        "actual_injection_state": {
            "state": "request_submitted",
            "skill_ids": list(expected),
        },
        "request_hash": request_hash,
        "failure_category": category,
        "runtime_identity": manifest["runtime_identity"],
        "answer_code_bundle_sha256": manifest["answer_code_bundle_sha256"],
        "decision_source_sha256": decision_source_sha256,
        "engine_attempts": engine_attempts,
        "error": dict(error_payload),
    }


def success_record(
    instance: JsonObject,
    arm: SemanticArm,
    model: str,
    model_label: str,
    expected: Sequence[str],
    request_hash: str,
    result: EngineResult,
    engine_attempts: int,
    manifest: RuntimeManifest,
    decision_source_sha256: str,
) -> JsonObject:
    """Build one successful answer record without altering model output."""

    record: JsonObject = {
        "schema_version": ANSWER_RECORD_SCHEMA_VERSION,
        "instance_id": require_string(
            instance.get("instance_id"),
            "answer.instance_id",
        ),
        "dataset": require_string(instance.get("dataset"), "answer.dataset"),
        "method": arm,
        "model": model_label,
        "served_model": model,
        "raw_output": result.raw_output,
        "skill_ids_used": list(result.skill_ids_used),
        "expected_skill_ids": list(expected),
        "actual_injection_state": {
            "state": "confirmed_by_engine",
            "skill_ids": list(result.skill_ids_used),
        },
        "request_hash": request_hash,
        "failure_category": "success",
        "runtime_identity": manifest["runtime_identity"],
        "answer_code_bundle_sha256": manifest["answer_code_bundle_sha256"],
        "decision_source_sha256": decision_source_sha256,
        "engine_attempts": engine_attempts,
    }
    if result.transcript is not None:
        record["transcript"] = result.transcript
    if result.meta:
        record["meta"] = dict(result.meta)
    return record


def run_one(
    instance: JsonObject,
    skills: list[JsonObject],
    expected: tuple[str, ...],
    request_hash: str,
    arm: SemanticArm,
    engine: DirectEngine,
    client: OpenAI,
    model: str,
    model_label: str,
    manifest: RuntimeManifest,
    decision_source_sha256: str,
    attempt_log: Path,
    request_error_types: tuple[type[Exception], ...],
    write_lock: threading.Lock,
) -> RunOutcome:
    """Run one answer with bounded infrastructure retries."""

    instance_id: str = require_string(
        instance.get("instance_id"),
        "answer.instance_id",
    )
    for engine_attempt in range(1, MAX_INFRA_ATTEMPTS + 1):
        started_at: float = time.time()
        try:
            result: EngineResult = engine.run(
                instance,
                skills,
                client,
                model,
            )
            append_attempt(
                attempt_log,
                {
                    "instance_id": instance_id,
                    "request_hash": request_hash,
                    "engine_attempt": engine_attempt,
                    "status": "response",
                    "elapsed_seconds": round(time.time() - started_at, 6),
                    "raw_output_empty": not bool(result.raw_output.strip()),
                },
                write_lock,
            )
            if not result.raw_output.strip():
                should_retry_empty: bool = (
                    engine_attempt < MAX_INFRA_ATTEMPTS
                )
                if should_retry_empty:
                    print(
                        canonical_json(
                            {
                                "level": "warning",
                                "event": "answer_empty_retry",
                                "model": model,
                                "instance_id": instance_id,
                                "engine_attempt": engine_attempt,
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(RETRY_DELAYS_SECONDS[engine_attempt - 1])
                    continue
                record: JsonObject = failure_record(
                    instance,
                    arm,
                    model,
                    model_label,
                    expected,
                    request_hash,
                    "method_failure",
                    {
                        "exception_name": "EmptyModelOutput",
                        "message": (
                            "Direct engine returned empty raw_output for all "
                            "bounded attempts"
                        ),
                        "status_code": None,
                        "response_body": "",
                    },
                    engine_attempt,
                    manifest,
                    decision_source_sha256,
                )
                return {"record": record, "engine_attempts": engine_attempt}
            record = success_record(
                instance,
                arm,
                model,
                model_label,
                expected,
                request_hash,
                result,
                engine_attempt,
                manifest,
                decision_source_sha256,
            )
            return {"record": record, "engine_attempts": engine_attempt}
        except request_error_types as error:
            exception_name, status_code, response_body = error_context(error)
            category: FailureCategory = classify_request_error(
                exception_name,
                str(error),
                status_code,
                response_body,
            )
            append_attempt(
                attempt_log,
                {
                    "instance_id": instance_id,
                    "request_hash": request_hash,
                    "engine_attempt": engine_attempt,
                    "status": "error",
                    "failure_category": category,
                    "exception_name": exception_name,
                    "message": str(error),
                    "status_code": status_code,
                    "response_body": response_body,
                    "elapsed_seconds": round(time.time() - started_at, 6),
                },
                write_lock,
            )
            should_retry: bool = (
                category == "infra_transient"
                and engine_attempt < MAX_INFRA_ATTEMPTS
            )
            if should_retry:
                print(
                    canonical_json(
                        {
                            "level": "warning",
                            "event": "answer_infra_retry",
                            "model": model,
                            "instance_id": instance_id,
                            "engine_attempt": engine_attempt,
                            "status_code": status_code,
                            "response_body": response_body,
                            "exception_name": exception_name,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(RETRY_DELAYS_SECONDS[engine_attempt - 1])
                continue
            record = failure_record(
                instance,
                arm,
                model,
                model_label,
                expected,
                request_hash,
                category,
                {
                    "exception_name": exception_name,
                    "message": str(error),
                    "status_code": status_code,
                    "response_body": response_body,
                },
                engine_attempt,
                manifest,
                decision_source_sha256,
            )
            return {"record": record, "engine_attempts": engine_attempt}
    raise AssertionError("Answer retry loop exited without a result")


def main() -> None:
    """Execute only audit-approved pending instances."""

    args = parse_args()
    workers: int = int(args.workers)
    if workers <= 0:
        raise ValueError(f"workers must be positive: workers={workers}")
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    decision_source_path: Path = cast(Path, args.decision_source).resolve()
    audit_path: Path = cast(Path, args.audit).resolve()
    preseed_path: Path = cast(Path, args.preseed).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    attempt_log_path: Path = cast(Path, args.attempt_log).resolve()
    manifest_path: Path = cast(Path, args.runtime_manifest).resolve()
    model: str = str(args.model)
    api_base: str = str(args.api_base)
    arm: SemanticArm = cast(SemanticArm, str(args.arm))
    domain: str = str(args.domain)

    runtime: AnswerRuntime = load_answer_runtime()
    engine_runtime: AnswerEngineRuntime = load_engine_runtime()
    instances: list[JsonObject] = load_instances(instances_path, domain)
    instance_index: dict[str, JsonObject] = {
        require_string(instance.get("instance_id"), "instance.instance_id"): instance
        for instance in instances
    }
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    decisions: dict[str, JsonObject] = load_decisions(decision_source_path)
    audit: dict[str, AuditRow] = load_audit(audit_path, arm)
    for name, observed_ids in (
        ("decision-source", list(decisions)),
        ("reuse-audit", list(audit)),
    ):
        coverage = audit_record_coverage(list(instance_index), observed_ids)
        if not coverage.complete:
            raise DownstreamDataError(
                f"{name} coverage mismatch: "
                f"missing={list(coverage.missing_ids)[:20]}, "
                f"unexpected={list(coverage.unexpected_ids)[:20]}"
            )
    manifest: RuntimeManifest = load_manifest(
        manifest_path,
        sha256_file(instances_path),
        sha256_file(corpus_path),
    )
    if manifest["runtime_identity"].get("served_model") != model:
        raise DownstreamDataError(
            "Answer model does not match runtime manifest: "
            f"cli_model={model}, "
            f"manifest_served_model={manifest['runtime_identity'].get('served_model')!r}"
        )
    decision_source_sha256: str = sha256_file(decision_source_path)
    for instance_id, instance in instance_index.items():
        expected: tuple[str, ...] = expected_skill_ids(
            arm,
            decisions[instance_id],
        )
        if list(expected) != audit[instance_id]["expected_skill_ids"]:
            raise DownstreamDataError(
                "Reuse audit expected decision is stale: "
                f"instance_id={instance_id}, decision_source={list(expected)}, "
                f"audit={audit[instance_id]['expected_skill_ids']}"
            )
        skills: list[JsonObject] = loaded_skills(
            expected,
            corpus,
            instance_id,
        )
        request_hash: str = answer_hash(
            arm,
            instance,
            skills,
            manifest,
            runtime,
        )
        if request_hash != audit[instance_id]["new_request_hash"]:
            raise DownstreamDataError(
                "Reuse audit request hash is stale: "
                f"instance_id={instance_id}, expected={request_hash}, "
                f"audit={audit[instance_id]['new_request_hash']}"
            )

    write_lock: threading.Lock = threading.Lock()
    initialize_output(output_path, preseed_path, audit, write_lock)
    existing_index: dict[str, AnswerLine] = rewrite_retryable_output(
        output_path,
        audit,
    )
    unexpected_output_ids: list[str] = sorted(
        set(existing_index) - set(instance_index)
    )
    if unexpected_output_ids:
        raise DownstreamDataError(
            "Answer output contains unexpected instances: "
            f"sample={unexpected_output_ids[:20]}"
        )
    pending_ids: list[str] = [
        instance_id
        for instance_id, audit_row in audit.items()
        if audit_row["needs_inference"] and instance_id not in existing_index
    ]
    client: OpenAI = engine_runtime["create_client"](api_base, None)
    engine: DirectEngine = engine_runtime["create_engine"](
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        thinking=THINKING,
    )
    model_label: str = engine_runtime["model_short_name"](model)

    futures: dict[Future[RunOutcome], str] = {}
    if pending_ids:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for instance_id in pending_ids:
                instance: JsonObject = instance_index[instance_id]
                expected: tuple[str, ...] = tuple(
                    audit[instance_id]["expected_skill_ids"]
                )
                skills: list[JsonObject] = loaded_skills(
                    expected,
                    corpus,
                    instance_id,
                )
                futures[
                    executor.submit(
                        run_one,
                        instance,
                        skills,
                        expected,
                        audit[instance_id]["new_request_hash"],
                        arm,
                        engine,
                        client,
                        model,
                        model_label,
                        manifest,
                        decision_source_sha256,
                        attempt_log_path,
                        engine_runtime["request_error_types"],
                        write_lock,
                    )
                ] = instance_id
            completed: int = 0
            for future in as_completed(futures):
                outcome: RunOutcome = future.result()
                append_raw_line(
                    output_path,
                    canonical_json(outcome["record"]),
                    write_lock,
                )
                completed += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        canonical_json(
                            {
                                "event": "answer_progress",
                                "model": model,
                                "domain": domain,
                                "arm": arm,
                                "completed_this_run": completed,
                                "pending_this_run": len(futures),
                            }
                        ),
                        flush=True,
                    )

    final_lines: list[AnswerLine] = load_jsonl(output_path, "answer-output")
    final_index: dict[str, AnswerLine] = indexed_answer_lines(
        final_lines,
        "answer-output",
    )
    final_coverage = audit_record_coverage(
        list(instance_index),
        list(final_index),
    )
    category_counts: dict[str, int] = {}
    reused_count: int = 0
    for instance_id, line in final_index.items():
        if audit[instance_id]["status"] == "reused_same_arm":
            reused_count += 1
            category: str = "success"
        else:
            category = validate_failure_category(
                line["record"].get("failure_category")
            )
        category_counts[category] = category_counts.get(category, 0) + 1
    summary: JsonObject = {
        "event": "k2_answer_job_complete",
        "model": model,
        "domain": domain,
        "arm": arm,
        "expected": len(instances),
        "observed": len(final_lines),
        "missing": len(final_coverage.missing_ids),
        "duplicates": len(final_coverage.duplicate_ids),
        "unexpected": len(final_coverage.unexpected_ids),
        "reused_same_arm": reused_count,
        "new_records": len(final_lines) - reused_count,
        "failure_categories": category_counts,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    print(canonical_json(summary), flush=True)
    unresolved: int = (
        category_counts.get("infra_transient", 0)
        + category_counts.get("unclassified_error", 0)
    )
    if not final_coverage.complete or unresolved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
