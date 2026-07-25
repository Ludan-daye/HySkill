#!/usr/bin/env python3
"""Answer fresh BM25 Select decisions with the K=2 native direct engine."""

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
from typing import Literal, Protocol, TypeAlias, TypedDict, cast

from hyskill.runtime_matched_execution import (
    ExecutionContext,
    JobBoundManifest,
    JsonLike,
    OpenAIClientLike,
    answer_payload_hash,
    bind_execution_context,
    canonical_json,
    classify_request_error,
    error_context,
    execution_request_hash,
    load_job_bound_manifest,
    manifest_artifact,
    sha256_file,
    validate_frozen_k2_runtime_reference,
    verify_job_bound_manifest_files,
    wrap_openai_client,
)
from hyskill.runtime_matched_select import (
    SELECT_ARM,
    SELECT_STAGE,
    SELECTION_RECORD_SCHEMA_VERSION,
    FailureCategory,
    JsonObject,
    JsonValue,
    SelectProtocolError,
    SelectionRecord,
    require_list,
    require_object,
    require_select_eligible,
    require_string,
    selected_skill_ids,
)


ANSWER_SCHEMA_VERSION: str = "runtime-matched-baseline-answer-v1"
ANSWER_PAYLOAD_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-answer-payload-v1"
)
ANSWER_STAGE: str = "answer"
ZERO_CALL_PAYLOAD_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-zero-call-payload-v1"
)
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 2048
THINKING: bool = False
MAX_INFRA_ATTEMPTS: int = 3
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0)
AnswerCategory: TypeAlias = Literal[
    "success",
    "method_failure",
    "infra_transient",
    "unclassified_error",
]
AttemptPayload: TypeAlias = Mapping[str, JsonLike]


class EngineResult(Protocol):
    """Fields returned by the native SR-Agents direct engine."""

    raw_output: str
    transcript: str | None
    skill_ids_used: list[str]


class DirectEngine(Protocol):
    """Native SR-Agents direct engine contract."""

    def run(
        self,
        instance: JsonObject,
        skills: list[JsonObject],
        client: object,
        model: str,
    ) -> EngineResult:
        """Execute the unchanged direct engine."""


class DirectEngineFactory(Protocol):
    """Native direct-engine constructor contract."""

    def __call__(
        self,
        *,
        temperature: float,
        max_tokens: int,
        thinking: bool,
    ) -> DirectEngine:
        """Create the frozen native direct engine."""


class CreateClient(Protocol):
    """SR-Agents OpenAI-compatible client factory contract."""

    def __call__(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> object:
        """Create one endpoint client."""


class BuildPrompt(Protocol):
    """SR-Agents dataset prompt builder contract."""

    def __call__(
        self,
        instance: JsonObject,
        skills: list[str] | None,
    ) -> tuple[str, str]:
        """Return the native system and user messages."""


class GetExtraBody(Protocol):
    """SR-Agents thinking-control helper contract."""

    def __call__(
        self,
        model: str,
        thinking: bool,
    ) -> JsonObject | None:
        """Return model-specific request additions."""


class AnswerRuntime(TypedDict):
    """Native direct-answer components loaded from SR-Agents."""

    create_client: CreateClient
    create_engine: DirectEngineFactory
    build_prompt: BuildPrompt
    get_extra_body: GetExtraBody
    request_error_types: tuple[type[Exception], ...]


class PreparedAnswer(TypedDict):
    """One validated answer payload and execution identity."""

    instance: JsonObject
    skills: list[JsonObject]
    expected_skill_ids: list[str]
    messages: list[JsonObject]
    tools: list[JsonObject]
    generation: JsonObject
    answer_payload_hash: str
    execution_request_hash: str
    decision_failure: JsonObject | None


class AnswerRecord(TypedDict):
    """One fresh runtime-matched baseline answer record."""

    schema_version: str
    instance_id: str
    model: str
    served_model: str
    domain: str
    arm: str
    stage: str
    raw_output: str
    transcript: str | None
    expected_skill_ids: list[str]
    skill_ids_used: list[str]
    actual_injection_state: JsonObject
    answer_payload_hash: str
    execution_request_hash: str
    failure_category: AnswerCategory
    runtime_manifest_sha256: str
    code_bundle_sha256: str
    decision_source_sha256: str
    answer_call_attempts: int
    reused_same_arm: bool
    error: JsonObject | None


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-domain Select answer job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt-log", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--max-new-records", required=True, type=int)
    return parser.parse_args()


def load_native_answer_runtime() -> AnswerRuntime:
    """Load the exact K=2 direct-answer implementation."""

    try:
        direct_module: ModuleType = importlib.import_module(
            "sragents.infer.engines.direct"
        )
        llm_module: ModuleType = importlib.import_module("sragents.llm")
        prompts_module: ModuleType = importlib.import_module(
            "sragents.prompts"
        )
        openai_module: ModuleType = importlib.import_module("openai")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The frozen SR-Agents answer runtime is unavailable. Install "
            "external/SR-Agents at revision 277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f "
            "inside the project environment before answering."
        ) from error
    api_error_type: object = getattr(openai_module, "APIError", None)
    if not isinstance(api_error_type, type) or not issubclass(
        api_error_type,
        Exception,
    ):
        raise RuntimeError(
            "The installed OpenAI client does not expose APIError"
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


def load_json(path: Path, context: str) -> JsonValue:
    """Load one UTF-8 JSON file."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    try:
        return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise SelectProtocolError(
            f"{context} JSON is malformed: path={path}, "
            f"line={error.lineno}, column={error.colno}, message={error.msg}"
        ) from error


def index_rows(
    rows: Sequence[JsonObject],
    context: str,
) -> dict[str, JsonObject]:
    """Index JSON rows by unique instance ID."""

    output: dict[str, JsonObject] = {}
    for row_number, row in enumerate(rows, start=1):
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}:{row_number}.instance_id",
        )
        if instance_id in output:
            raise SelectProtocolError(
                f"{context} contains duplicate instance ID: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def object_rows(
    values: Sequence[JsonValue],
    context: str,
) -> list[JsonObject]:
    """Validate a sequence of JSON object rows."""

    return [
        require_object(value, f"{context}[{index}]")
        for index, value in enumerate(values)
    ]


def load_instances(
    path: Path,
    domain: str,
    expected_count: int,
) -> list[JsonObject]:
    """Load one exact-domain instance set."""

    rows: list[JsonObject] = object_rows(
        require_list(load_json(path, "instances"), "instances"),
        "instances",
    )
    if len(rows) != expected_count:
        raise SelectProtocolError(
            "Instance denominator mismatch: "
            f"expected={expected_count}, actual={len(rows)}, path={path}"
        )
    index_rows(rows, "instances")
    for instance in rows:
        instance_id: str = require_string(
            instance.get("instance_id"),
            "instance.instance_id",
        )
        instance_domain: str = require_string(
            instance.get("dataset"),
            f"instance:{instance_id}.dataset",
        )
        if instance_domain != domain:
            raise SelectProtocolError(
                "Instance domain mismatch: "
                f"instance_id={instance_id}, expected={domain}, "
                f"actual={instance_domain}"
            )
    return rows


def load_corpus(path: Path) -> dict[str, JsonObject]:
    """Load a duplicate-free frozen corpus."""

    rows: list[JsonObject] = object_rows(
        require_list(load_json(path, "corpus"), "corpus"),
        "corpus",
    )
    output: dict[str, JsonObject] = {}
    for row_number, row in enumerate(rows, start=1):
        skill_id: str = require_string(
            row.get("skill_id"),
            f"corpus:{row_number}.skill_id",
        )
        if skill_id in output:
            raise SelectProtocolError(
                f"Corpus contains duplicate skill ID: skill_id={skill_id}"
            )
        output[skill_id] = row
    if not output:
        raise SelectProtocolError(f"Corpus is empty: path={path}")
    return output


def load_decisions(
    path: Path,
    expected_ids: set[str],
    result_tag: str,
    served_model: str,
    domain: str,
) -> dict[str, SelectionRecord]:
    """Load one complete fresh Select decision file."""

    if not path.is_file():
        raise FileNotFoundError(f"Decision file does not exist: path={path}")
    rows: list[SelectionRecord] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise SelectProtocolError(
                    "Decision JSONL is malformed: "
                    f"path={path}, line={line_number}, "
                    f"column={error.colno}, message={error.msg}"
                ) from error
            row: JsonObject = require_object(
                raw_value,
                f"decisions:{path}:{line_number}",
            )
            if row.get("schema_version") != SELECTION_RECORD_SCHEMA_VERSION:
                raise SelectProtocolError(
                    "Decision schema mismatch: "
                    f"path={path}, line={line_number}, "
                    f"schema={row.get('schema_version')!r}"
                )
            expected_fields: tuple[tuple[str, str], ...] = (
                ("model", result_tag),
                ("served_model", served_model),
                ("domain", domain),
                ("arm", SELECT_ARM),
                ("stage", SELECT_STAGE),
            )
            for field_name, expected_value in expected_fields:
                actual_value: str = require_string(
                    row.get(field_name),
                    f"decision:{line_number}.{field_name}",
                )
                if actual_value != expected_value:
                    raise SelectProtocolError(
                        "Decision identity mismatch: "
                        f"line={line_number}, field={field_name}, "
                        f"expected={expected_value}, actual={actual_value}"
                    )
            if row.get("reused_same_arm") is not False:
                raise SelectProtocolError(
                    "Fresh Select decision unexpectedly reuses a legacy row: "
                    f"line={line_number}, "
                    f"reused_same_arm={row.get('reused_same_arm')!r}"
                )
            selected_skill_ids(row)
            rows.append(cast(SelectionRecord, row))
    index: dict[str, JsonObject] = index_rows(
        cast(list[JsonObject], rows),
        "decisions",
    )
    observed_ids: set[str] = set(index)
    if observed_ids != expected_ids:
        raise SelectProtocolError(
            "Decision coverage mismatch: "
            f"missing={sorted(expected_ids - observed_ids)[:20]}, "
            f"unexpected={sorted(observed_ids - expected_ids)[:20]}"
        )
    return cast(dict[str, SelectionRecord], index)


def manifest_object(
    value: JsonValue | None,
    context: str,
) -> JsonObject:
    """Return one manifest object using the local strict JSON type."""

    return require_object(value, context)


def validate_manifest_binding(
    manifest: JobBoundManifest,
    instances_path: Path,
    corpus_path: Path,
    decisions_path: Path,
    result_tag: str,
    served_model: str,
    domain: str,
    api_base: str,
    runtime: AnswerRuntime,
) -> str:
    """Validate the answer manifest against this exact job."""

    facts: JsonObject = manifest_object(
        cast(JsonValue, manifest["runtime_facts"]),
        "manifest.runtime_facts",
    )
    job: JsonObject = manifest_object(
        facts.get("job"),
        "manifest.runtime_facts.job",
    )
    expected_job_fields: tuple[tuple[str, str], ...] = (
        ("result_tag", result_tag),
        ("model", served_model),
        ("domain", domain),
        ("arm", SELECT_ARM),
        ("stage", ANSWER_STAGE),
    )
    for field_name, expected_value in expected_job_fields:
        actual_value: str = require_string(
            job.get(field_name),
            f"manifest.job.{field_name}",
        )
        if actual_value != expected_value:
            raise SelectProtocolError(
                "Runtime manifest job mismatch: "
                f"field={field_name}, expected={expected_value}, "
                f"actual={actual_value}"
            )
    endpoint: JsonObject = manifest_object(
        facts.get("endpoint"),
        "manifest.runtime_facts.endpoint",
    )
    endpoint_api_base: str = require_string(
        endpoint.get("api_base"),
        "manifest.endpoint.api_base",
    )
    if endpoint_api_base.rstrip("/") != api_base.rstrip("/"):
        raise SelectProtocolError(
            "CLI API base differs from the job-bound endpoint: "
            f"cli={api_base}, manifest={endpoint_api_base}"
        )
    endpoint_model: str = require_string(
        endpoint.get("served_model"),
        "manifest.endpoint.served_model",
    )
    if endpoint_model != served_model:
        raise SelectProtocolError(
            "CLI served model differs from endpoint manifest: "
            f"cli={served_model}, manifest={endpoint_model}"
        )
    for artifact_name, path in (
        ("instances", instances_path),
        ("corpus", corpus_path),
        ("select_decisions", decisions_path),
    ):
        evidence = manifest_artifact(manifest, artifact_name)
        if Path(evidence["path"]).resolve() != path.resolve():
            raise SelectProtocolError(
                "Manifest artifact path mismatch: "
                f"name={artifact_name}, expected={path.resolve()}, "
                f"actual={evidence['path']}"
            )
        actual_size: int = path.stat().st_size
        actual_sha256: str = sha256_file(path)
        if (
            evidence["size_bytes"] != actual_size
            or evidence["sha256"] != actual_sha256
        ):
            raise SelectProtocolError(
                "Manifest artifact identity mismatch: "
                f"name={artifact_name}, expected_size={evidence['size_bytes']}, "
                f"actual_size={actual_size}, expected_sha={evidence['sha256']}, "
                f"actual_sha={actual_sha256}"
            )
    expected_generation: JsonObject = {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "extra_body": runtime["get_extra_body"](served_model, THINKING),
    }
    manifest_generation: JsonObject = manifest_object(
        cast(JsonValue, manifest["generation"]),
        "manifest.generation",
    )
    for field_name, expected_value in expected_generation.items():
        if manifest_generation.get(field_name) != expected_value:
            raise SelectProtocolError(
                "Answer generation differs from the job manifest: "
                f"field={field_name}, expected={expected_value!r}, "
                f"actual={manifest_generation.get(field_name)!r}"
            )
    return require_string(job.get("job_id"), "manifest.job.job_id")


def loaded_skills(
    skill_ids: Sequence[str],
    corpus: Mapping[str, JsonObject],
    instance_id: str,
) -> list[JsonObject]:
    """Resolve one frozen selection to exact corpus rows."""

    missing_ids: list[str] = [
        skill_id for skill_id in skill_ids if skill_id not in corpus
    ]
    if missing_ids:
        raise SelectProtocolError(
            "Selected skills are absent from the frozen corpus: "
            f"instance_id={instance_id}, missing={missing_ids}"
        )
    return [corpus[skill_id] for skill_id in skill_ids]


def rendered_messages(
    instance: JsonObject,
    skills: Sequence[JsonObject],
    build_prompt: BuildPrompt,
) -> list[JsonObject]:
    """Render the exact initial messages used by the native direct engine."""

    skill_texts: list[str] = [
        cast(str, skill["content"])
        for skill in skills
        if isinstance(skill.get("content"), str)
        and cast(str, skill["content"])
    ]
    system, user = build_prompt(instance, skill_texts)
    messages: list[JsonObject] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def rendered_tools(skills: Sequence[JsonObject]) -> list[JsonObject]:
    """Flatten native skill tools without modifying their order."""

    output: list[JsonObject] = []
    for skill_index, skill in enumerate(skills):
        raw_tools: JsonValue | None = skill.get("tools", [])
        tools: list[JsonValue] = require_list(
            raw_tools,
            f"skills[{skill_index}].tools",
        )
        output.extend(
            require_object(
                tool,
                f"skills[{skill_index}].tools[{tool_index}]",
            )
            for tool_index, tool in enumerate(tools)
        )
    return output


def prepare_answer(
    instance: JsonObject,
    decision: SelectionRecord,
    corpus: Mapping[str, JsonObject],
    runtime: AnswerRuntime,
    served_model: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
) -> PreparedAnswer:
    """Build one normal answer payload or deterministic zero-call payload."""

    instance_id: str = require_string(
        instance.get("instance_id"),
        "instance.instance_id",
    )
    skill_ids: tuple[str, ...] = selected_skill_ids(decision)
    skills: list[JsonObject] = loaded_skills(
        skill_ids,
        corpus,
        instance_id,
    )
    generation: JsonObject = {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "extra_body": runtime["get_extra_body"](served_model, THINKING),
    }
    decision_category: FailureCategory = decision["failure_category"]
    if decision_category == "method_failure":
        messages: list[JsonObject] = []
        tools: list[JsonObject] = []
        payload_schema: str = ZERO_CALL_PAYLOAD_SCHEMA_VERSION
        decision_error: JsonObject = {
            "exception_name": "SelectDecisionMethodFailure",
            "message": (
                "Answer request was not submitted because the native "
                "50-candidate selector had a method failure."
            ),
            "decision_failure_category": decision_category,
            "decision_execution_request_hash": decision[
                "execution_request_hash"
            ],
        }
    else:
        messages = rendered_messages(
            instance,
            skills,
            runtime["build_prompt"],
        )
        tools = rendered_tools(skills)
        payload_schema = ANSWER_PAYLOAD_SCHEMA_VERSION
        decision_error = None
    payload_sha256: str = answer_payload_hash(
        payload_schema,
        instance,
        messages,
        skills,
        tools,
        generation,
    )
    return {
        "instance": instance,
        "skills": skills,
        "expected_skill_ids": list(skill_ids),
        "messages": messages,
        "tools": tools,
        "generation": generation,
        "answer_payload_hash": payload_sha256,
        "execution_request_hash": execution_request_hash(
            payload_schema,
            payload_sha256,
            runtime_manifest_sha256,
            code_bundle_sha256,
        ),
        "decision_failure": decision_error,
    }


def answer_record(
    prepared: PreparedAnswer,
    result_tag: str,
    served_model: str,
    domain: str,
    raw_output: str,
    transcript: str | None,
    skill_ids_used: Sequence[str],
    category: AnswerCategory,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    decision_source_sha256: str,
    answer_call_attempts: int,
    error: JsonObject | None,
) -> AnswerRecord:
    """Build one immutable fresh answer row."""

    instance_id: str = require_string(
        prepared["instance"].get("instance_id"),
        "prepared.instance_id",
    )
    if category == "success":
        if not raw_output.strip():
            raise SelectProtocolError(
                f"Successful answer is empty: instance_id={instance_id}"
            )
        if list(skill_ids_used) != prepared["expected_skill_ids"]:
            raise SelectProtocolError(
                "Native engine used skills different from the decision: "
                f"instance_id={instance_id}, "
                f"expected={prepared['expected_skill_ids']}, "
                f"actual={list(skill_ids_used)}"
            )
        injection_state: JsonObject = {
            "state": "confirmed_by_engine",
            "skill_ids": list(skill_ids_used),
        }
    elif prepared["decision_failure"] is not None:
        if answer_call_attempts != 0:
            raise SelectProtocolError(
                "Decision-stage failure unexpectedly made answer calls: "
                f"instance_id={instance_id}, calls={answer_call_attempts}"
            )
        injection_state = {
            "state": "decision_failed_zero_call",
            "skill_ids": [],
        }
    else:
        injection_state = {
            "state": "request_submitted",
            "skill_ids": list(prepared["expected_skill_ids"]),
        }
    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "instance_id": instance_id,
        "model": result_tag,
        "served_model": served_model,
        "domain": domain,
        "arm": SELECT_ARM,
        "stage": ANSWER_STAGE,
        "raw_output": raw_output,
        "transcript": transcript,
        "expected_skill_ids": list(prepared["expected_skill_ids"]),
        "skill_ids_used": list(skill_ids_used),
        "actual_injection_state": injection_state,
        "answer_payload_hash": prepared["answer_payload_hash"],
        "execution_request_hash": prepared["execution_request_hash"],
        "failure_category": category,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "code_bundle_sha256": code_bundle_sha256,
        "decision_source_sha256": decision_source_sha256,
        "answer_call_attempts": answer_call_attempts,
        "reused_same_arm": False,
        "error": error,
    }


def append_jsonl(
    path: Path,
    payload: AttemptPayload,
    write_lock: threading.Lock,
) -> None:
    """Append and flush one canonical JSONL event."""

    line: str = canonical_json(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with write_lock:
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(line)
            output_file.flush()


def write_records_atomic(
    path: Path,
    records: Sequence[AnswerRecord],
) -> None:
    """Atomically replace one answer JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(canonical_json(record))
            output_file.write("\n")
    temporary_path.replace(path)


def read_records(path: Path) -> list[AnswerRecord]:
    """Read existing fresh answer records for strict resume."""

    if not path.exists():
        return []
    records: list[AnswerRecord] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise SelectProtocolError(
                    "Answer JSONL is malformed: "
                    f"path={path}, line={line_number}, "
                    f"column={error.colno}, message={error.msg}"
                ) from error
            row: JsonObject = require_object(
                raw_value,
                f"answers:{path}:{line_number}",
            )
            if row.get("schema_version") != ANSWER_SCHEMA_VERSION:
                raise SelectProtocolError(
                    "Existing answer has unexpected schema: "
                    f"path={path}, line={line_number}, "
                    f"schema={row.get('schema_version')!r}"
                )
            instance_id: str = require_string(
                row.get("instance_id"),
                f"answers:{path}:{line_number}.instance_id",
            )
            if instance_id in seen_ids:
                raise SelectProtocolError(
                    "Answer output contains duplicate instance: "
                    f"instance_id={instance_id}"
                )
            seen_ids.add(instance_id)
            records.append(cast(AnswerRecord, row))
    return records


def validate_existing_record(
    record: AnswerRecord,
    prepared: PreparedAnswer,
    result_tag: str,
    served_model: str,
    domain: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    decision_source_sha256: str,
) -> None:
    """Reject stale or cross-job resume answers."""

    expected_fields: tuple[tuple[str, object], ...] = (
        ("model", result_tag),
        ("served_model", served_model),
        ("domain", domain),
        ("arm", SELECT_ARM),
        ("stage", ANSWER_STAGE),
        ("answer_payload_hash", prepared["answer_payload_hash"]),
        (
            "execution_request_hash",
            prepared["execution_request_hash"],
        ),
        ("runtime_manifest_sha256", runtime_manifest_sha256),
        ("code_bundle_sha256", code_bundle_sha256),
        ("decision_source_sha256", decision_source_sha256),
        ("reused_same_arm", False),
    )
    for field_name, expected_value in expected_fields:
        actual_value: object = record.get(field_name)
        if actual_value != expected_value:
            raise SelectProtocolError(
                "Existing answer is stale or belongs to another job: "
                f"instance_id={record.get('instance_id')}, "
                f"field={field_name}, expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )


def main() -> None:
    """Run one resumable fresh Select answer job."""

    args = parse_args()
    workers: int = int(args.workers)
    expected_count: int = int(args.expected_count)
    max_new_records: int = int(args.max_new_records)
    if workers <= 0:
        raise ValueError(f"workers must be positive: workers={workers}")
    if expected_count <= 0:
        raise ValueError(
            f"expected-count must be positive: value={expected_count}"
        )
    if max_new_records < 0:
        raise ValueError(
            "max-new-records must be zero or positive: "
            f"value={max_new_records}"
        )
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    decisions_path: Path = cast(Path, args.decisions).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    attempt_log_path: Path = cast(Path, args.attempt_log).resolve()
    manifest_path: Path = cast(Path, args.runtime_manifest).resolve()
    repository_root: Path = cast(Path, args.repository_root).resolve()
    result_tag: str = str(args.result_tag)
    served_model: str = str(args.model)
    api_base: str = str(args.api_base)
    domain: str = str(args.domain)

    require_select_eligible(result_tag)
    runtime: AnswerRuntime = load_native_answer_runtime()
    instances: list[JsonObject] = load_instances(
        instances_path,
        domain,
        expected_count,
    )
    instance_index: dict[str, JsonObject] = index_rows(
        instances,
        "instances",
    )
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    decisions: dict[str, SelectionRecord] = load_decisions(
        decisions_path,
        set(instance_index),
        result_tag,
        served_model,
        domain,
    )
    manifest: JobBoundManifest = load_job_bound_manifest(manifest_path)
    validate_frozen_k2_runtime_reference(manifest["runtime_facts"])
    verify_job_bound_manifest_files(manifest, repository_root)
    job_id: str = validate_manifest_binding(
        manifest,
        instances_path,
        corpus_path,
        decisions_path,
        result_tag,
        served_model,
        domain,
        api_base,
        runtime,
    )
    runtime_manifest_sha256: str = sha256_file(manifest_path)
    code_bundle_sha256: str = manifest["code_bundle_sha256"]
    decision_source_sha256: str = sha256_file(decisions_path)
    prepared_answers: list[PreparedAnswer] = [
        prepare_answer(
            instance,
            decisions[
                require_string(
                    instance.get("instance_id"),
                    "instance.instance_id",
                )
            ],
            corpus,
            runtime,
            served_model,
            runtime_manifest_sha256,
            code_bundle_sha256,
        )
        for instance in instances
    ]
    prepared_index: dict[str, PreparedAnswer] = {
        require_string(
            prepared["instance"].get("instance_id"),
            "prepared.instance_id",
        ): prepared
        for prepared in prepared_answers
    }
    existing_records: list[AnswerRecord] = read_records(output_path)
    retained_records: list[AnswerRecord] = [
        record
        for record in existing_records
        if record.get("failure_category") != "infra_transient"
    ]
    if len(retained_records) != len(existing_records):
        write_records_atomic(output_path, retained_records)
    existing_index: dict[str, AnswerRecord] = {}
    for record in retained_records:
        instance_id: str = record["instance_id"]
        if instance_id not in prepared_index:
            raise SelectProtocolError(
                "Existing answer has unexpected instance: "
                f"instance_id={instance_id}"
            )
        validate_existing_record(
            record,
            prepared_index[instance_id],
            result_tag,
            served_model,
            domain,
            runtime_manifest_sha256,
            code_bundle_sha256,
            decision_source_sha256,
        )
        existing_index[instance_id] = record

    write_lock: threading.Lock = threading.Lock()

    def usage_sink(event: Mapping[str, JsonLike]) -> None:
        append_jsonl(attempt_log_path, event, write_lock)

    raw_client: object = runtime["create_client"](api_base, None)
    client: object = wrap_openai_client(
        cast(OpenAIClientLike, raw_client),
        usage_sink,
    )
    engine: DirectEngine = runtime["create_engine"](
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        thinking=THINKING,
    )

    def run_one(prepared: PreparedAnswer) -> AnswerRecord:
        instance_id: str = require_string(
            prepared["instance"].get("instance_id"),
            "prepared.instance_id",
        )
        if prepared["decision_failure"] is not None:
            return answer_record(
                prepared,
                result_tag,
                served_model,
                domain,
                "",
                None,
                (),
                "method_failure",
                runtime_manifest_sha256,
                code_bundle_sha256,
                decision_source_sha256,
                0,
                prepared["decision_failure"],
            )
        for answer_attempt in range(1, MAX_INFRA_ATTEMPTS + 1):
            context: ExecutionContext = ExecutionContext(
                job_id,
                served_model,
                domain,
                SELECT_ARM,
                instance_id,
                answer_attempt,
                prepared["answer_payload_hash"],
                prepared["execution_request_hash"],
            )
            started_at: float = time.monotonic()
            try:
                with bind_execution_context(context):
                    result: EngineResult = engine.run(
                        prepared["instance"],
                        prepared["skills"],
                        client,
                        served_model,
                    )
                append_jsonl(
                    attempt_log_path,
                    {
                        "schema_version": (
                            "runtime-matched-answer-attempt-v1"
                        ),
                        "job_id": job_id,
                        "model": served_model,
                        "domain": domain,
                        "arm": SELECT_ARM,
                        "instance_id": instance_id,
                        "answer_attempt": answer_attempt,
                        "status": "response",
                        "answer_payload_hash": prepared[
                            "answer_payload_hash"
                        ],
                        "execution_request_hash": prepared[
                            "execution_request_hash"
                        ],
                        "elapsed_seconds": round(
                            time.monotonic() - started_at,
                            6,
                        ),
                        "raw_output_empty": not bool(
                            result.raw_output.strip()
                        ),
                    },
                    write_lock,
                )
                if not result.raw_output.strip():
                    if answer_attempt < MAX_INFRA_ATTEMPTS:
                        print(
                            canonical_json(
                                {
                                    "level": "warning",
                                    "event": "answer_empty_retry",
                                    "job_id": job_id,
                                    "model": served_model,
                                    "domain": domain,
                                    "instance_id": instance_id,
                                    "answer_attempt": answer_attempt,
                                }
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(
                            RETRY_DELAYS_SECONDS[answer_attempt - 1]
                        )
                        continue
                    return answer_record(
                        prepared,
                        result_tag,
                        served_model,
                        domain,
                        "",
                        result.transcript,
                        (),
                        "method_failure",
                        runtime_manifest_sha256,
                        code_bundle_sha256,
                        decision_source_sha256,
                        answer_attempt,
                        {
                            "exception_name": "EmptyModelOutput",
                            "message": (
                                "Direct engine returned empty raw_output for "
                                "all bounded attempts."
                            ),
                            "status_code": None,
                            "response_body": "",
                        },
                    )
                return answer_record(
                    prepared,
                    result_tag,
                    served_model,
                    domain,
                    result.raw_output,
                    result.transcript,
                    result.skill_ids_used,
                    "success",
                    runtime_manifest_sha256,
                    code_bundle_sha256,
                    decision_source_sha256,
                    answer_attempt,
                    None,
                )
            except runtime["request_error_types"] as error:
                details = error_context(error)
                category: FailureCategory = classify_request_error(
                    details.exception_name,
                    details.message,
                    details.status_code,
                    details.response_body,
                )
                append_jsonl(
                    attempt_log_path,
                    {
                        "schema_version": (
                            "runtime-matched-answer-attempt-v1"
                        ),
                        "job_id": job_id,
                        "model": served_model,
                        "domain": domain,
                        "arm": SELECT_ARM,
                        "instance_id": instance_id,
                        "answer_attempt": answer_attempt,
                        "status": "error",
                        "failure_category": category,
                        "answer_payload_hash": prepared[
                            "answer_payload_hash"
                        ],
                        "execution_request_hash": prepared[
                            "execution_request_hash"
                        ],
                        "exception_name": details.exception_name,
                        "message": details.message,
                        "status_code": details.status_code,
                        "response_body": details.response_body,
                        "elapsed_seconds": round(
                            time.monotonic() - started_at,
                            6,
                        ),
                    },
                    write_lock,
                )
                should_retry: bool = (
                    category == "infra_transient"
                    and answer_attempt < MAX_INFRA_ATTEMPTS
                )
                if should_retry:
                    print(
                        canonical_json(
                            {
                                "level": "warning",
                                "event": "answer_infra_retry",
                                "job_id": job_id,
                                "model": served_model,
                                "domain": domain,
                                "instance_id": instance_id,
                                "answer_attempt": answer_attempt,
                                "exception_name": details.exception_name,
                                "status_code": details.status_code,
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(
                        RETRY_DELAYS_SECONDS[answer_attempt - 1]
                    )
                    continue
                answer_category: AnswerCategory = cast(
                    AnswerCategory,
                    category,
                )
                return answer_record(
                    prepared,
                    result_tag,
                    served_model,
                    domain,
                    "",
                    None,
                    (),
                    answer_category,
                    runtime_manifest_sha256,
                    code_bundle_sha256,
                    decision_source_sha256,
                    answer_attempt,
                    {
                        "exception_name": details.exception_name,
                        "message": details.message,
                        "status_code": details.status_code,
                        "response_body": details.response_body,
                    },
                )
        raise AssertionError(
            "Answer retry loop exited without a result"
        )

    pending_ids: list[str] = sorted(
        set(prepared_index) - set(existing_index)
    )
    selected_ids: list[str] = (
        pending_ids
        if max_new_records == 0
        else pending_ids[:max_new_records]
    )
    pending: list[PreparedAnswer] = [
        prepared_index[instance_id] for instance_id in selected_ids
    ]
    if pending:
        futures: dict[Future[AnswerRecord], str] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for prepared in pending:
                instance_id: str = require_string(
                    prepared["instance"].get("instance_id"),
                    "prepared.instance_id",
                )
                futures[executor.submit(run_one, prepared)] = instance_id
            completed: int = 0
            for future in as_completed(futures):
                record: AnswerRecord = future.result()
                append_jsonl(
                    output_path,
                    cast(AttemptPayload, record),
                    write_lock,
                )
                existing_index[record["instance_id"]] = record
                completed += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        canonical_json(
                            {
                                "event": "select_answer_progress",
                                "job_id": job_id,
                                "model": served_model,
                                "domain": domain,
                                "completed_this_run": completed,
                                "pending_this_run": len(futures),
                                "total_records": len(existing_index),
                            }
                        ),
                        flush=True,
                    )

    final_records: list[AnswerRecord] = read_records(output_path)
    final_ids: list[str] = [
        record["instance_id"] for record in final_records
    ]
    missing_ids: list[str] = sorted(set(instance_index) - set(final_ids))
    unexpected_ids: list[str] = sorted(
        set(final_ids) - set(instance_index)
    )
    duplicate_count: int = len(final_ids) - len(set(final_ids))
    category_counts: dict[str, int] = {}
    zero_call_method_failures: int = 0
    for record in final_records:
        category: str = record["failure_category"]
        category_counts[category] = category_counts.get(category, 0) + 1
        injection_state: JsonObject = record["actual_injection_state"]
        if injection_state.get("state") == "decision_failed_zero_call":
            zero_call_method_failures += 1
    unresolved: int = (
        category_counts.get("infra_transient", 0)
        + category_counts.get("unclassified_error", 0)
    )
    summary: JsonObject = {
        "event": "runtime_matched_select_answers_complete",
        "job_id": job_id,
        "model": served_model,
        "domain": domain,
        "run_mode": "full" if max_new_records == 0 else "canary",
        "expected": expected_count,
        "observed": len(final_records),
        "selected_this_run": len(selected_ids),
        "missing": len(missing_ids),
        "unexpected": len(unexpected_ids),
        "duplicates": duplicate_count,
        "reused_same_arm": 0,
        "zero_call_decision_method_failures": zero_call_method_failures,
        "failure_categories": category_counts,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "attempt_log": str(attempt_log_path),
    }
    print(canonical_json(summary), flush=True)
    full_run: bool = max_new_records == 0
    if (
        unexpected_ids
        or duplicate_count
        or unresolved
        or (
            full_run
            and (
                len(final_records) != expected_count
                or missing_ids
            )
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
