#!/usr/bin/env python3
"""Run one fresh runtime-matched Bare answer job."""

from __future__ import annotations

import argparse
import importlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict, cast

from hyskill.runtime_matched_execution import (
    ExecutionContext,
    FailureCategory,
    JobBoundManifest,
    JsonLike,
    JsonObject,
    JsonValue,
    OpenAIClientLike,
    RuntimeMatchedExecutionError,
    RuntimeManifestError,
    UsageSink,
    answer_payload_hash,
    bind_execution_context,
    canonical_json,
    classify_request_error,
    error_context,
    execution_request_hash,
    load_job_bound_manifest,
    load_json_file,
    manifest_artifact,
    require_json_object,
    require_nonempty_string,
    sha256_file,
    verify_job_bound_manifest_files,
    validate_frozen_k2_runtime_reference,
    wrap_openai_client,
)


ANSWER_RECORD_SCHEMA_VERSION: str = "runtime-matched-baseline-answer-v1"
ANSWER_PAYLOAD_SCHEMA_VERSION: str = "runtime-matched-answer-payload-v1"
BARE_ARM: str = "bare"
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 2048
THINKING: bool = False
MAX_ENGINE_ATTEMPTS: int = 3
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0)


class BareExecutionError(RuntimeMatchedExecutionError):
    """Raised when a Bare job violates its frozen execution contract."""


class EngineResult(Protocol):
    """Fields returned by the native SR-Agents direct engine."""

    raw_output: str
    transcript: str | None
    skill_ids_used: list[str]
    meta: dict[str, object]


class DirectEngine(Protocol):
    """Native SR-Agents direct-engine contract."""

    def run(
        self,
        instance: JsonObject,
        skills: list[JsonObject],
        client: object,
        model: str,
    ) -> EngineResult:
        """Run one direct answer."""


class DirectEngineFactory(Protocol):
    """Native direct-engine constructor contract."""

    def __call__(
        self,
        *,
        temperature: float,
        max_tokens: int,
        thinking: bool,
    ) -> DirectEngine:
        """Create one frozen direct engine."""


class CreateClient(Protocol):
    """SR-Agents OpenAI-compatible client factory contract."""

    def __call__(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> object:
        """Create one endpoint client."""


class BuildPrompt(Protocol):
    """Native SR-Agents dataset prompt builder contract."""

    def __call__(
        self,
        instance: JsonObject,
        skills: list[str] | None,
    ) -> tuple[str, str]:
        """Return system and user prompt strings."""


class GetExtraBody(Protocol):
    """Native SR-Agents thinking-control helper contract."""

    def __call__(
        self,
        model: str,
        thinking: bool,
    ) -> JsonObject | None:
        """Return model-specific request additions."""


class NativeBareRuntime(TypedDict):
    """Exact native components used by the Bare runner."""

    create_client: CreateClient
    create_engine: DirectEngineFactory
    build_prompt: BuildPrompt
    get_extra_body: GetExtraBody
    request_error_types: tuple[type[Exception], ...]


class PreparedBareRequest(TypedDict):
    """One fully rendered and hashed Bare request."""

    instance: JsonObject
    messages: list[JsonObject]
    answer_payload_hash: str
    execution_request_hash: str


class JsonlLine(TypedDict):
    """One parsed JSONL row and its source line."""

    raw_line: str
    record: JsonObject


class BareRunOutcome(TypedDict):
    """One terminal answer outcome."""

    record: JsonObject
    engine_attempts: int


class BareJobSummary(TypedDict):
    """Machine-readable state after one canary or full invocation."""

    event: str
    job_id: str
    model: str
    served_model: str
    domain: str
    arm: str
    run_mode: str
    expected_total: int
    observed_total: int
    pending_before_run: int
    selected_this_run: int
    completed_this_run: int
    unresolved: int
    missing_after_run: int
    failure_categories: dict[str, int]
    reused_same_arm: int
    runtime_manifest_sha256: str
    output: str
    output_sha256: str
    run_valid: bool


class JsonlUsageSink:
    """Thread-safe append-only persistence for one usage journal."""

    def __init__(self, path: Path, write_lock: threading.Lock) -> None:
        self._path: Path = path
        self._write_lock: threading.Lock = write_lock

    def __call__(self, event: Mapping[str, JsonLike]) -> None:
        append_jsonl(self._path, event, self._write_lock)


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-domain Bare job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--usage-log", required=True, type=Path)
    parser.add_argument("--attempt-log", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--max-new-records", required=True, type=int)
    return parser.parse_args()


def load_native_bare_runtime() -> NativeBareRuntime:
    """Load the unchanged native direct-answer implementation."""

    try:
        direct_module: ModuleType = importlib.import_module(
            "sragents.infer.engines.direct"
        )
        llm_module: ModuleType = importlib.import_module("sragents.llm")
        prompts_module: ModuleType = importlib.import_module("sragents.prompts")
        openai_module: ModuleType = importlib.import_module("openai")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Runtime-matched Bare dependencies are unavailable. Install the "
            "project environment and the pinned external/SR-Agents checkout "
            "at revision 277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f."
        ) from error
    api_error_type: object = getattr(openai_module, "APIError", None)
    if not isinstance(api_error_type, type) or not issubclass(
        api_error_type,
        Exception,
    ):
        raise RuntimeError(
            "Installed OpenAI client does not expose APIError for bounded "
            "request-failure handling"
        )
    return {
        "create_client": cast(
            CreateClient,
            getattr(llm_module, "create_llm_client"),
        ),
        "create_engine": cast(
            DirectEngineFactory,
            getattr(direct_module, "DirectEngine"),
        ),
        "build_prompt": cast(
            BuildPrompt,
            getattr(prompts_module, "build_prompt"),
        ),
        "get_extra_body": cast(
            GetExtraBody,
            getattr(llm_module, "get_extra_body"),
        ),
        "request_error_types": (cast(type[Exception], api_error_type),),
    }


def append_jsonl(
    path: Path,
    payload: Mapping[str, JsonLike],
    write_lock: threading.Lock,
) -> None:
    """Append one canonical JSONL record and flush it before returning."""

    line: str = canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with write_lock:
        with path.open("a", encoding="utf-8", newline="") as output_file:
            output_file.write(line)
            output_file.write("\n")
            output_file.flush()


def load_jsonl(
    path: Path,
    context: str,
    allow_missing: bool,
) -> list[JsonlLine]:
    """Load strict JSONL rows while preserving exact source lines."""

    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"{context} JSONL does not exist: path={path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"{context} JSONL is not a regular file: path={path}"
        )
    output: list[JsonlLine] = []
    with path.open(encoding="utf-8", newline="") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                raise BareExecutionError(
                    f"{context} JSONL contains a blank line: "
                    f"path={path}, line={line_number}"
                )
            try:
                value: JsonValue = cast(JsonValue, json.loads(raw_line))
            except json.JSONDecodeError as error:
                raise BareExecutionError(
                    f"{context} JSONL is malformed: path={path}, "
                    f"line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            output.append(
                {
                    "raw_line": raw_line,
                    "record": require_json_object(
                        value,
                        f"{context}:{path}:{line_number}",
                    ),
                }
            )
    return output


def load_instances(path: Path, domain: str) -> list[JsonObject]:
    """Load unique instances for exactly one declared rule domain."""

    value: JsonValue = load_json_file(path, "instances")
    if not isinstance(value, list) or not value:
        raise BareExecutionError(
            f"Instances must be a non-empty JSON list: path={path}"
        )
    instances: list[JsonObject] = []
    observed_ids: set[str] = set()
    for index, raw_instance in enumerate(value):
        instance: JsonObject = require_json_object(
            raw_instance,
            f"instances[{index}]",
        )
        instance_id: str = require_nonempty_string(
            instance.get("instance_id"),
            f"instances[{index}].instance_id",
        )
        if instance_id in observed_ids:
            raise BareExecutionError(
                f"Instances contain a duplicate ID: instance_id={instance_id}"
            )
        observed_ids.add(instance_id)
        observed_domain: str = require_nonempty_string(
            instance.get("dataset"),
            f"instances[{index}].dataset",
        )
        if observed_domain != domain:
            raise BareExecutionError(
                "Instance domain differs from the job domain: "
                f"instance_id={instance_id}, expected={domain}, "
                f"observed={observed_domain}"
            )
        instances.append(instance)
    return instances


def required_nested_object(
    parent: Mapping[str, JsonValue],
    key: str,
    context: str,
) -> JsonObject:
    """Return one required nested JSON object."""

    return require_json_object(parent.get(key), f"{context}.{key}")


def validate_named_artifact(
    manifest: JobBoundManifest,
    name: str,
    observed_path: Path,
) -> str:
    """Bind one CLI input path to one exact manifest artifact."""

    artifact = manifest_artifact(manifest, name)
    expected_path: Path = Path(artifact["path"]).resolve()
    if expected_path != observed_path:
        raise RuntimeManifestError(
            "CLI input path differs from the job-bound manifest: "
            f"name={name}, expected={expected_path}, observed={observed_path}"
        )
    observed_sha256: str = sha256_file(observed_path)
    if observed_sha256 != artifact["sha256"]:
        raise RuntimeManifestError(
            "CLI input SHA differs from the job-bound manifest: "
            f"name={name}, expected={artifact['sha256']}, "
            f"observed={observed_sha256}"
        )
    return observed_sha256


def validate_bare_manifest(
    manifest: JobBoundManifest,
    repository_root: Path,
    instances_path: Path,
    corpus_path: Path,
    result_tag: str,
    model: str,
    api_base: str,
    domain: str,
    runtime: NativeBareRuntime,
) -> tuple[str, str]:
    """Validate the complete job identity and frozen Bare protocol."""

    verify_job_bound_manifest_files(manifest, repository_root)
    facts: JsonObject = manifest["runtime_facts"]
    job: JsonObject = required_nested_object(facts, "job", "runtime_facts")
    endpoint: JsonObject = required_nested_object(
        facts,
        "endpoint",
        "runtime_facts",
    )
    validate_frozen_k2_runtime_reference(facts)
    expected_job_fields: tuple[tuple[str, JsonValue], ...] = (
        ("result_tag", result_tag),
        ("model", model),
        ("domain", domain),
        ("arm", BARE_ARM),
    )
    mismatches: list[str] = [
        f"{field_name}:expected={expected!r},observed={job.get(field_name)!r}"
        for field_name, expected in expected_job_fields
        if job.get(field_name) != expected
    ]
    expected_endpoint_fields: tuple[tuple[str, JsonValue], ...] = (
        ("api_base", api_base),
        ("served_model", model),
        ("dtype", "bfloat16"),
        ("quantization", "none"),
        ("max_model_len", 8192),
    )
    mismatches.extend(
        f"endpoint.{field_name}:expected={expected!r},"
        f"observed={endpoint.get(field_name)!r}"
        for field_name, expected in expected_endpoint_fields
        if endpoint.get(field_name) != expected
    )
    native_extra_body: JsonObject | None = runtime["get_extra_body"](
        model,
        THINKING,
    )
    expected_generation: JsonObject = {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "extra_body": cast(JsonValue, native_extra_body),
    }
    for field_name, expected in expected_generation.items():
        if manifest["generation"].get(field_name) != expected:
            mismatches.append(
                f"generation.{field_name}:expected={expected!r},"
                f"observed={manifest['generation'].get(field_name)!r}"
            )
    unexpected_generation_fields: list[str] = sorted(
        set(manifest["generation"]) - set(expected_generation)
    )
    if unexpected_generation_fields:
        mismatches.append(
            "generation.unexpected_fields:"
            f"value={unexpected_generation_fields}"
        )
    if mismatches:
        raise RuntimeManifestError(
            "Bare runtime manifest does not match the frozen job: "
            f"mismatches={mismatches}"
        )
    instances_sha256: str = validate_named_artifact(
        manifest,
        "instances",
        instances_path,
    )
    corpus_sha256: str = validate_named_artifact(
        manifest,
        "corpus",
        corpus_path,
    )
    return instances_sha256, corpus_sha256


def rendered_messages(
    instance: JsonObject,
    runtime: NativeBareRuntime,
) -> list[JsonObject]:
    """Render exactly the messages submitted by native DirectEngine."""

    system, user = runtime["build_prompt"](instance, [])
    if not isinstance(system, str) or not isinstance(user, str):
        raise BareExecutionError(
            "Native build_prompt must return two strings: "
            f"instance_id={instance.get('instance_id')!r}, "
            f"system_type={type(system).__name__}, "
            f"user_type={type(user).__name__}"
        )
    messages: list[JsonObject] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def prepare_bare_request(
    instance: JsonObject,
    generation: JsonObject,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    runtime: NativeBareRuntime,
) -> PreparedBareRequest:
    """Render and hash one model-visible Bare request."""

    messages: list[JsonObject] = rendered_messages(instance, runtime)
    payload_sha256: str = answer_payload_hash(
        ANSWER_PAYLOAD_SCHEMA_VERSION,
        instance,
        messages,
        [],
        [],
        generation,
    )
    return {
        "instance": instance,
        "messages": messages,
        "answer_payload_hash": payload_sha256,
        "execution_request_hash": execution_request_hash(
            ANSWER_PAYLOAD_SCHEMA_VERSION,
            payload_sha256,
            runtime_manifest_sha256,
            code_bundle_sha256,
        ),
    }


def indexed_lines(
    lines: Sequence[JsonlLine],
    context: str,
) -> dict[str, JsonlLine]:
    """Index JSONL rows by a required unique instance ID."""

    output: dict[str, JsonlLine] = {}
    for index, line in enumerate(lines):
        instance_id: str = require_nonempty_string(
            line["record"].get("instance_id"),
            f"{context}[{index}].instance_id",
        )
        if instance_id in output:
            raise BareExecutionError(
                f"{context} contains a duplicate instance: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = line
    return output


def validate_failure_category(value: JsonValue | None) -> FailureCategory:
    """Return one supported terminal answer category."""

    if value not in (
        "success",
        "infra_transient",
        "method_failure",
        "unclassified_error",
    ):
        raise BareExecutionError(
            f"Bare answer has an invalid failure category: value={value!r}"
        )
    return cast(FailureCategory, value)


def validate_existing_outputs(
    output_lines: Sequence[JsonlLine],
    prepared: Mapping[str, PreparedBareRequest],
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    result_tag: str,
    model: str,
    domain: str,
) -> dict[str, JsonlLine]:
    """Validate every preserved final-bound output before resuming."""

    output_index: dict[str, JsonlLine] = indexed_lines(
        output_lines,
        "answer-output",
    )
    unexpected_ids: list[str] = sorted(set(output_index) - set(prepared))
    if unexpected_ids:
        raise BareExecutionError(
            "Answer output contains IDs outside the job instances: "
            f"sample={unexpected_ids[:20]}"
        )
    job: JsonObject = required_nested_object(
        manifest["runtime_facts"],
        "job",
        "runtime_facts",
    )
    for instance_id, line in output_index.items():
        record: JsonObject = line["record"]
        expected_fields: tuple[tuple[str, JsonValue], ...] = (
            ("schema_version", ANSWER_RECORD_SCHEMA_VERSION),
            ("stage", "answer"),
            ("job_id", job["job_id"]),
            ("dataset", domain),
            ("domain", domain),
            ("arm", BARE_ARM),
            ("model", result_tag),
            ("served_model", model),
            (
                "answer_payload_hash",
                prepared[instance_id]["answer_payload_hash"],
            ),
            (
                "execution_request_hash",
                prepared[instance_id]["execution_request_hash"],
            ),
            ("runtime_manifest_sha256", runtime_manifest_sha256),
            ("code_bundle_sha256", manifest["code_bundle_sha256"]),
            ("reused_same_arm", False),
        )
        mismatches: list[str] = [
            f"{field_name}:expected={expected!r},"
            f"observed={record.get(field_name)!r}"
            for field_name, expected in expected_fields
            if record.get(field_name) != expected
        ]
        if mismatches:
            raise BareExecutionError(
                "Existing Bare output is stale or belongs to another job: "
                f"instance_id={instance_id}, mismatches={mismatches}"
            )
        category: FailureCategory = validate_failure_category(
            record.get("failure_category")
        )
        raw_output: JsonValue | None = record.get("raw_output")
        if not isinstance(raw_output, str):
            raise BareExecutionError(
                "Existing Bare raw_output must be a string: "
                f"instance_id={instance_id}, value_type={type(raw_output).__name__}"
            )
        if category == "success" and not raw_output.strip():
            raise BareExecutionError(
                "Successful Bare output is empty: "
                f"instance_id={instance_id}"
            )
        if category != "success" and raw_output:
            raise BareExecutionError(
                "Failed Bare output must not carry model answer text: "
                f"instance_id={instance_id}, category={category}"
            )
    return output_index


def validate_usage_journal(
    usage_lines: Sequence[JsonlLine],
    output_index: Mapping[str, JsonlLine],
    prepared: Mapping[str, PreparedBareRequest],
    job_id: str,
    model: str,
    domain: str,
) -> None:
    """Reject stale, duplicate, or orphan request evidence before resume."""

    observed_event_ids: set[tuple[str, int, int]] = set()
    usage_instance_ids: set[str] = set()
    for index, line in enumerate(usage_lines):
        event: JsonObject = line["record"]
        instance_id: str = require_nonempty_string(
            event.get("instance_id"),
            f"usage[{index}].instance_id",
        )
        if instance_id not in prepared:
            raise BareExecutionError(
                "Usage journal contains an instance outside this job: "
                f"instance_id={instance_id}"
            )
        logical_attempt: JsonValue | None = event.get("logical_attempt")
        http_subcall: JsonValue | None = event.get("http_subcall")
        if (
            isinstance(logical_attempt, bool)
            or not isinstance(logical_attempt, int)
            or logical_attempt <= 0
            or isinstance(http_subcall, bool)
            or not isinstance(http_subcall, int)
            or http_subcall <= 0
        ):
            raise BareExecutionError(
                "Usage journal attempt identifiers must be positive integers: "
                f"instance_id={instance_id}, "
                f"logical_attempt={logical_attempt!r}, "
                f"http_subcall={http_subcall!r}"
            )
        event_id: tuple[str, int, int] = (
            instance_id,
            logical_attempt,
            http_subcall,
        )
        if event_id in observed_event_ids:
            raise BareExecutionError(
                f"Usage journal contains a duplicate event: event_id={event_id}"
            )
        observed_event_ids.add(event_id)
        expected_fields: tuple[tuple[str, JsonValue], ...] = (
            ("schema_version", "runtime-matched-usage-event-v1"),
            ("job_id", job_id),
            ("model", model),
            ("domain", domain),
            ("arm", BARE_ARM),
            (
                "answer_payload_hash",
                prepared[instance_id]["answer_payload_hash"],
            ),
            (
                "execution_request_hash",
                prepared[instance_id]["execution_request_hash"],
            ),
        )
        mismatches: list[str] = [
            f"{field_name}:expected={expected!r},"
            f"observed={event.get(field_name)!r}"
            for field_name, expected in expected_fields
            if event.get(field_name) != expected
        ]
        if mismatches:
            raise BareExecutionError(
                "Usage journal event is stale or misattributed: "
                f"event_id={event_id}, mismatches={mismatches}"
            )
        usage_instance_ids.add(instance_id)
    orphan_ids: list[str] = sorted(usage_instance_ids - set(output_index))
    if orphan_ids:
        raise BareExecutionError(
            "Usage journal contains completed HTTP calls without terminal "
            "answer rows. Refusing to silently resample stochastic requests: "
            f"orphan_instance_ids={orphan_ids[:20]}, count={len(orphan_ids)}"
        )
    missing_usage_ids: list[str] = sorted(set(output_index) - usage_instance_ids)
    if missing_usage_ids:
        raise BareExecutionError(
            "Existing answer rows have no bound HTTP usage evidence: "
            f"instance_ids={missing_usage_ids[:20]}, "
            f"count={len(missing_usage_ids)}"
        )


def attempt_payload(
    context: ExecutionContext,
    status: str,
    elapsed_seconds: float,
    details: Mapping[str, JsonLike],
) -> JsonObject:
    """Build one logical engine-attempt journal record."""

    payload: JsonObject = {
        "schema_version": "runtime-matched-answer-attempt-v1",
        "job_id": context.job_id,
        "model": context.model,
        "domain": context.domain,
        "arm": context.arm,
        "instance_id": context.instance_id,
        "logical_attempt": context.logical_attempt,
        "answer_payload_hash": context.answer_payload_hash,
        "execution_request_hash": context.execution_request_hash,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }
    payload.update(cast(dict[str, JsonValue], dict(details)))
    return payload


def base_answer_record(
    prepared: PreparedBareRequest,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    instances_sha256: str,
    corpus_sha256: str,
    result_tag: str,
    model: str,
    domain: str,
    engine_attempts: int,
) -> JsonObject:
    """Build fields shared by successful and failed Bare records."""

    job: JsonObject = required_nested_object(
        manifest["runtime_facts"],
        "job",
        "runtime_facts",
    )
    instance_id: str = require_nonempty_string(
        prepared["instance"].get("instance_id"),
        "answer.instance_id",
    )
    return {
        "schema_version": ANSWER_RECORD_SCHEMA_VERSION,
        "stage": "answer",
        "job_id": job["job_id"],
        "instance_id": instance_id,
        "dataset": domain,
        "domain": domain,
        "arm": BARE_ARM,
        "method": BARE_ARM,
        "model": result_tag,
        "served_model": model,
        "skill_ids_used": [],
        "actual_injection_state": {
            "state": "bare",
            "skill_ids": [],
        },
        "answer_payload_hash": prepared["answer_payload_hash"],
        "execution_request_hash": prepared["execution_request_hash"],
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "code_bundle_sha256": manifest["code_bundle_sha256"],
        "instances_sha256": instances_sha256,
        "corpus_sha256": corpus_sha256,
        "generation": manifest["generation"],
        "engine_attempts": engine_attempts,
        "reused_same_arm": False,
    }


def success_record(
    prepared: PreparedBareRequest,
    result: EngineResult,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    instances_sha256: str,
    corpus_sha256: str,
    result_tag: str,
    model: str,
    domain: str,
    engine_attempts: int,
) -> JsonObject:
    """Build one successful fresh Bare answer record."""

    record: JsonObject = base_answer_record(
        prepared,
        manifest,
        runtime_manifest_sha256,
        instances_sha256,
        corpus_sha256,
        result_tag,
        model,
        domain,
        engine_attempts,
    )
    record.update(
        {
            "raw_output": result.raw_output,
            "failure_category": "success",
        }
    )
    if result.transcript is not None:
        record["transcript"] = result.transcript
    if result.meta:
        record["meta"] = cast(JsonObject, dict(result.meta))
    return record


def failure_record(
    prepared: PreparedBareRequest,
    category: FailureCategory,
    error_payload: Mapping[str, JsonLike],
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    instances_sha256: str,
    corpus_sha256: str,
    result_tag: str,
    model: str,
    domain: str,
    engine_attempts: int,
) -> JsonObject:
    """Build one terminal Bare method or request failure."""

    record: JsonObject = base_answer_record(
        prepared,
        manifest,
        runtime_manifest_sha256,
        instances_sha256,
        corpus_sha256,
        result_tag,
        model,
        domain,
        engine_attempts,
    )
    record.update(
        {
            "raw_output": "",
            "failure_category": category,
            "error": cast(JsonObject, dict(error_payload)),
        }
    )
    return record


def run_one(
    prepared: PreparedBareRequest,
    engine: DirectEngine,
    client: object,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    instances_sha256: str,
    corpus_sha256: str,
    result_tag: str,
    model: str,
    domain: str,
    attempt_log_path: Path,
    request_error_types: tuple[type[Exception], ...],
    attempt_write_lock: threading.Lock,
    max_engine_attempts: int,
    retry_delays_seconds: tuple[float, ...],
) -> BareRunOutcome:
    """Run one Bare answer with bounded request and empty-output retries."""

    if max_engine_attempts <= 0:
        raise ValueError(
            "max_engine_attempts must be positive: "
            f"value={max_engine_attempts}"
        )
    if len(retry_delays_seconds) != max_engine_attempts - 1:
        raise ValueError(
            "retry_delays_seconds must provide one delay per retry: "
            f"attempts={max_engine_attempts}, delays={retry_delays_seconds}"
        )
    instance_id: str = require_nonempty_string(
        prepared["instance"].get("instance_id"),
        "answer.instance_id",
    )
    job: JsonObject = required_nested_object(
        manifest["runtime_facts"],
        "job",
        "runtime_facts",
    )
    job_id: str = require_nonempty_string(job.get("job_id"), "job.job_id")
    for engine_attempt in range(1, max_engine_attempts + 1):
        context = ExecutionContext(
            job_id,
            model,
            domain,
            BARE_ARM,
            instance_id,
            engine_attempt,
            prepared["answer_payload_hash"],
            prepared["execution_request_hash"],
        )
        started_at: float = time.monotonic()
        try:
            with bind_execution_context(context):
                result: EngineResult = engine.run(
                    prepared["instance"],
                    [],
                    client,
                    model,
                )
        except request_error_types as error:
            details = error_context(error)
            category: FailureCategory = classify_request_error(
                details.exception_name,
                details.message,
                details.status_code,
                details.response_body,
            )
            append_jsonl(
                attempt_log_path,
                attempt_payload(
                    context,
                    "error",
                    time.monotonic() - started_at,
                    {
                        "failure_category": category,
                        "exception_name": details.exception_name,
                        "message": details.message,
                        "status_code": details.status_code,
                        "response_body": details.response_body,
                    },
                ),
                attempt_write_lock,
            )
            if (
                category == "infra_transient"
                and engine_attempt < max_engine_attempts
            ):
                time.sleep(retry_delays_seconds[engine_attempt - 1])
                continue
            record: JsonObject = failure_record(
                prepared,
                category,
                {
                    "exception_name": details.exception_name,
                    "message": details.message,
                    "status_code": details.status_code,
                    "response_body": details.response_body,
                },
                manifest,
                runtime_manifest_sha256,
                instances_sha256,
                corpus_sha256,
                result_tag,
                model,
                domain,
                engine_attempt,
            )
            return {"record": record, "engine_attempts": engine_attempt}
        if result.skill_ids_used:
            raise BareExecutionError(
                "Native direct engine reported injected skills for Bare: "
                f"instance_id={instance_id}, "
                f"skill_ids_used={result.skill_ids_used}"
            )
        raw_output: object = result.raw_output
        if not isinstance(raw_output, str):
            raise BareExecutionError(
                "Native direct engine returned non-string raw_output: "
                f"instance_id={instance_id}, "
                f"value_type={type(raw_output).__name__}"
            )
        append_jsonl(
            attempt_log_path,
            attempt_payload(
                context,
                "response",
                time.monotonic() - started_at,
                {
                    "raw_output_empty": not bool(raw_output.strip()),
                },
            ),
            attempt_write_lock,
        )
        if not raw_output.strip():
            if engine_attempt < max_engine_attempts:
                time.sleep(retry_delays_seconds[engine_attempt - 1])
                continue
            record = failure_record(
                prepared,
                "method_failure",
                {
                    "exception_name": "EmptyModelOutput",
                    "message": (
                        "Native direct engine returned empty output for all "
                        "bounded attempts"
                    ),
                    "status_code": None,
                    "response_body": "",
                },
                manifest,
                runtime_manifest_sha256,
                instances_sha256,
                corpus_sha256,
                result_tag,
                model,
                domain,
                engine_attempt,
            )
            return {"record": record, "engine_attempts": engine_attempt}
        record = success_record(
            prepared,
            result,
            manifest,
            runtime_manifest_sha256,
            instances_sha256,
            corpus_sha256,
            result_tag,
            model,
            domain,
            engine_attempt,
        )
        return {"record": record, "engine_attempts": engine_attempt}
    raise AssertionError("Bare retry loop exited without a terminal outcome")


def failure_category_counts(
    output_index: Mapping[str, JsonlLine],
) -> dict[str, int]:
    """Count terminal categories in a validated output index."""

    counts: dict[str, int] = {}
    for line in output_index.values():
        category: FailureCategory = validate_failure_category(
            line["record"].get("failure_category")
        )
        counts[category] = counts.get(category, 0) + 1
    return counts


def run_bare_job(
    instances_path: Path,
    corpus_path: Path,
    runtime_manifest_path: Path,
    repository_root: Path,
    output_path: Path,
    usage_log_path: Path,
    attempt_log_path: Path,
    result_tag: str,
    model: str,
    api_base: str,
    domain: str,
    workers: int,
    max_new_records: int,
    runtime: NativeBareRuntime,
    max_engine_attempts: int,
    retry_delays_seconds: tuple[float, ...],
) -> BareJobSummary:
    """Run a canary slice or all remaining rows for one Bare job."""

    if workers <= 0:
        raise ValueError(f"workers must be positive: value={workers}")
    if max_new_records < 0:
        raise ValueError(
            "max_new_records must be zero or positive: "
            f"value={max_new_records}"
        )
    resolved_instances_path: Path = instances_path.resolve()
    resolved_corpus_path: Path = corpus_path.resolve()
    resolved_manifest_path: Path = runtime_manifest_path.resolve()
    resolved_repository_root: Path = repository_root.resolve()
    resolved_output_path: Path = output_path.resolve()
    resolved_usage_log_path: Path = usage_log_path.resolve()
    resolved_attempt_log_path: Path = attempt_log_path.resolve()
    journal_paths: tuple[Path, ...] = (
        resolved_output_path,
        resolved_usage_log_path,
        resolved_attempt_log_path,
    )
    if len(set(journal_paths)) != len(journal_paths):
        raise BareExecutionError(
            f"Output and journal paths must be distinct: paths={journal_paths}"
        )
    manifest: JobBoundManifest = load_job_bound_manifest(
        resolved_manifest_path
    )
    instances_sha256, corpus_sha256 = validate_bare_manifest(
        manifest,
        resolved_repository_root,
        resolved_instances_path,
        resolved_corpus_path,
        result_tag,
        model,
        api_base,
        domain,
        runtime,
    )
    runtime_manifest_sha256: str = sha256_file(resolved_manifest_path)
    instances: list[JsonObject] = load_instances(
        resolved_instances_path,
        domain,
    )
    prepared: dict[str, PreparedBareRequest] = {}
    for instance in instances:
        instance_id: str = require_nonempty_string(
            instance.get("instance_id"),
            "instance.instance_id",
        )
        prepared[instance_id] = prepare_bare_request(
            instance,
            manifest["generation"],
            runtime_manifest_sha256,
            manifest["code_bundle_sha256"],
            runtime,
        )
    output_lines: list[JsonlLine] = load_jsonl(
        resolved_output_path,
        "answer-output",
        True,
    )
    output_index: dict[str, JsonlLine] = validate_existing_outputs(
        output_lines,
        prepared,
        manifest,
        runtime_manifest_sha256,
        result_tag,
        model,
        domain,
    )
    usage_lines: list[JsonlLine] = load_jsonl(
        resolved_usage_log_path,
        "usage",
        True,
    )
    job: JsonObject = required_nested_object(
        manifest["runtime_facts"],
        "job",
        "runtime_facts",
    )
    job_id: str = require_nonempty_string(job.get("job_id"), "job.job_id")
    validate_usage_journal(
        usage_lines,
        output_index,
        prepared,
        job_id,
        model,
        domain,
    )
    pending_ids: list[str] = sorted(set(prepared) - set(output_index))
    selected_ids: list[str] = (
        pending_ids
        if max_new_records == 0
        else pending_ids[:max_new_records]
    )
    output_write_lock: threading.Lock = threading.Lock()
    usage_write_lock: threading.Lock = threading.Lock()
    attempt_write_lock: threading.Lock = threading.Lock()
    completed_this_run: int = 0
    if selected_ids:
        raw_client: object = runtime["create_client"](api_base, None)
        client: OpenAIClientLike = wrap_openai_client(
            cast(OpenAIClientLike, raw_client),
            cast(
                UsageSink,
                JsonlUsageSink(
                    resolved_usage_log_path,
                    usage_write_lock,
                ),
            ),
        )
        engine: DirectEngine = runtime["create_engine"](
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            thinking=THINKING,
        )
        futures: dict[Future[BareRunOutcome], str] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for instance_id in selected_ids:
                futures[
                    executor.submit(
                        run_one,
                        prepared[instance_id],
                        engine,
                        client,
                        manifest,
                        runtime_manifest_sha256,
                        instances_sha256,
                        corpus_sha256,
                        result_tag,
                        model,
                        domain,
                        resolved_attempt_log_path,
                        runtime["request_error_types"],
                        attempt_write_lock,
                        max_engine_attempts,
                        retry_delays_seconds,
                    )
                ] = instance_id
            for future in as_completed(futures):
                outcome: BareRunOutcome = future.result()
                append_jsonl(
                    resolved_output_path,
                    outcome["record"],
                    output_write_lock,
                )
                completed_this_run += 1
                if (
                    completed_this_run % 100 == 0
                    or completed_this_run == len(futures)
                ):
                    print(
                        canonical_json(
                            {
                                "event": "runtime_matched_bare_progress",
                                "job_id": job_id,
                                "model": result_tag,
                                "served_model": model,
                                "domain": domain,
                                "completed_this_run": completed_this_run,
                                "selected_this_run": len(selected_ids),
                            }
                        ),
                        flush=True,
                    )
    final_output_lines: list[JsonlLine] = load_jsonl(
        resolved_output_path,
        "answer-output",
        True,
    )
    final_output_index: dict[str, JsonlLine] = validate_existing_outputs(
        final_output_lines,
        prepared,
        manifest,
        runtime_manifest_sha256,
        result_tag,
        model,
        domain,
    )
    final_usage_lines: list[JsonlLine] = load_jsonl(
        resolved_usage_log_path,
        "usage",
        True,
    )
    validate_usage_journal(
        final_usage_lines,
        final_output_index,
        prepared,
        job_id,
        model,
        domain,
    )
    categories: dict[str, int] = failure_category_counts(final_output_index)
    unresolved: int = (
        categories.get("infra_transient", 0)
        + categories.get("unclassified_error", 0)
    )
    missing_after_run: int = len(set(prepared) - set(final_output_index))
    full_run: bool = max_new_records == 0
    run_valid: bool = (
        completed_this_run == len(selected_ids)
        and unresolved == 0
        and (not full_run or missing_after_run == 0)
    )
    return {
        "event": "runtime_matched_bare_job_state",
        "job_id": job_id,
        "model": result_tag,
        "served_model": model,
        "domain": domain,
        "arm": BARE_ARM,
        "run_mode": "full" if full_run else "canary",
        "expected_total": len(prepared),
        "observed_total": len(final_output_index),
        "pending_before_run": len(pending_ids),
        "selected_this_run": len(selected_ids),
        "completed_this_run": completed_this_run,
        "unresolved": unresolved,
        "missing_after_run": missing_after_run,
        "failure_categories": categories,
        "reused_same_arm": 0,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "output": str(resolved_output_path),
        "output_sha256": (
            sha256_file(resolved_output_path)
            if resolved_output_path.is_file()
            else ""
        ),
        "run_valid": run_valid,
    }


def main() -> None:
    """Execute one CLI Bare canary or full resume invocation."""

    args = parse_args()
    summary: BareJobSummary = run_bare_job(
        cast(Path, args.instances),
        cast(Path, args.corpus),
        cast(Path, args.runtime_manifest),
        cast(Path, args.repository_root),
        cast(Path, args.output),
        cast(Path, args.usage_log),
        cast(Path, args.attempt_log),
        str(args.result_tag),
        str(args.model),
        str(args.api_base),
        str(args.domain),
        int(args.workers),
        int(args.max_new_records),
        load_native_bare_runtime(),
        MAX_ENGINE_ATTEMPTS,
        RETRY_DELAYS_SECONDS,
    )
    print(canonical_json(summary), flush=True)
    if not summary["run_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
