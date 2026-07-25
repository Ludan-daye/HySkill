#!/usr/bin/env python3
"""Run the frozen SR-Agents LLM selector without generating task answers."""

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
    CandidateDisplay,
    DownstreamDataError,
    FailureCategory,
    JsonLike,
    JsonObject,
    JsonValue,
    RuntimeManifest,
    SelectorGeneration,
    audit_record_coverage,
    canonical_json,
    classify_request_error,
    selector_request_fingerprint,
    sha256_file,
    sha256_json,
    validate_failure_category,
    validate_selector_runtime_manifest,
)

if TYPE_CHECKING:
    from openai import OpenAI


SELECTOR_SCHEMA_VERSION: str = "k2-selector-request-v1"
SELECTION_RECORD_SCHEMA_VERSION: str = "k2-selection-record-v1"
POOL_SIZE: int = 50
MAX_PARSE_ATTEMPTS: int = 3
MAX_INFRA_ATTEMPTS: int = 3
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 64
THINKING: bool = False
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0)


class BuildPrompt(Protocol):
    """SR-Agents dataset prompt builder contract."""

    def __call__(
        self,
        instance: JsonObject,
        skills: list[str] | None,
    ) -> tuple[str, str]:
        """Build the system and user messages."""


class FormatCandidates(Protocol):
    """SR-Agents selector candidate formatter contract."""

    def __call__(self, candidates: list[JsonObject]) -> str:
        """Format ordered candidates exactly as the native selector."""


class ParseFirstNumber(Protocol):
    """SR-Agents selector parser contract."""

    def __call__(self, response: str, candidate_count: int) -> int | None:
        """Return a zero-based candidate index."""


class DisplayName(Protocol):
    """SR-Agents skill display-name contract."""

    def __call__(
        self,
        skill: JsonObject,
        index: int | None,
    ) -> str:
        """Return the non-leaking display name."""


class Chat(Protocol):
    """SR-Agents chat helper contract."""

    def __call__(
        self,
        client: OpenAI,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
        extra_body: JsonObject | None,
    ) -> str:
        """Run one chat request."""


class CreateClient(Protocol):
    """SR-Agents OpenAI-compatible client factory contract."""

    def __call__(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> OpenAI:
        """Create a shared endpoint client."""


class GetExtraBody(Protocol):
    """SR-Agents thinking-control helper contract."""

    def __call__(
        self,
        model: str,
        thinking: bool,
    ) -> JsonObject | None:
        """Return request body additions."""


class SelectorRuntime(TypedDict):
    """Exact native selector functions loaded from the installed SR-Agents."""

    prompt_template: str
    build_prompt: BuildPrompt
    format_candidates: FormatCandidates
    parse_first_number: ParseFirstNumber
    display_name: DisplayName
    chat: Chat
    create_client: CreateClient
    get_extra_body: GetExtraBody
    request_error_types: tuple[type[Exception], ...]


class SelectorInput(TypedDict):
    """Validated input for one logical selector decision."""

    instance: JsonObject
    source_record: JsonObject
    candidates: list[JsonObject]
    candidate_displays: list[CandidateDisplay]
    rendered_prompt: str
    candidate_hash: str
    request_hash: str


class SelectionRecord(TypedDict):
    """Persisted selector decision and request provenance."""

    schema_version: str
    instance_id: str
    dataset: str
    arm: str
    model: str
    ordered_candidate_ids: list[str]
    candidate_hash: str
    selector_request_hash: str
    selected_skill_id: str | None
    selected_rank: int | None
    raw_response: str
    raw_responses: list[str]
    parse_attempts: int
    client_call_attempts: int
    parse_success: bool
    rank1_fallback: bool
    failure_category: FailureCategory
    runtime_identity: dict[str, JsonValue]
    code_bundle_sha256: str
    source_sha256: str
    error: dict[str, JsonValue] | None


class SelectionOutcome(TypedDict):
    """Internal selector result before provenance fields are attached."""

    selected_rank_zero_based: int | None
    raw_responses: list[str]
    parse_attempts: int
    client_call_attempts: int
    parse_success: bool
    rank1_fallback: bool
    failure_category: FailureCategory
    error: dict[str, JsonValue] | None


class SelectionCallFailure(RuntimeError):
    """Request failure with a frozen failure category and request context."""

    def __init__(
        self,
        category: FailureCategory,
        exception_name: str,
        message: str,
        status_code: int | None,
        response_body: str,
        client_call_attempts: int,
    ) -> None:
        super().__init__(message)
        self.category: FailureCategory = category
        self.exception_name: str = exception_name
        self.status_code: int | None = status_code
        self.response_body: str = response_body
        self.client_call_attempts: int = client_call_attempts


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-domain selection job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-source", required=True, type=Path)
    parser.add_argument("--attempt-log", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--code-bundle-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--workers", required=True, type=int)
    return parser.parse_args()


def load_selector_runtime() -> SelectorRuntime:
    """Load the exact native selector implementation from SR-Agents."""

    try:
        selector_module: ModuleType = importlib.import_module(
            "sragents.infer.providers.llm_select"
        )
        prompts_module: ModuleType = importlib.import_module("sragents.prompts")
        corpus_module: ModuleType = importlib.import_module("sragents.corpus")
        llm_module: ModuleType = importlib.import_module("sragents.llm")
        openai_module: ModuleType = importlib.import_module("openai")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SR-Agents runtime is unavailable. Install the project environment "
            "before running selection-only: "
            "command=.venv/bin/pip install --no-deps -e external/SR-Agents; "
            "install any missing packages separately from an approved China mirror"
        ) from error
    prompt_template: JsonValue | None = cast(
        JsonValue | None,
        getattr(selector_module, "_PROMPT", None),
    )
    if not isinstance(prompt_template, str):
        raise RuntimeError(
            "Installed SR-Agents does not expose the frozen llm_select prompt: "
            "attribute=sragents.infer.providers.llm_select._PROMPT"
        )
    api_error_type: type[Exception] | None = cast(
        type[Exception] | None,
        getattr(openai_module, "APIError", None),
    )
    if api_error_type is None or not issubclass(api_error_type, Exception):
        raise RuntimeError(
            "Installed OpenAI client does not expose APIError for bounded retries"
        )
    return {
        "prompt_template": prompt_template,
        "build_prompt": cast(BuildPrompt, getattr(prompts_module, "build_prompt")),
        "format_candidates": cast(
            FormatCandidates, getattr(selector_module, "_format_candidates")
        ),
        "parse_first_number": cast(
            ParseFirstNumber, getattr(selector_module, "_parse_first_number")
        ),
        "display_name": cast(DisplayName, getattr(corpus_module, "display_name")),
        "chat": cast(Chat, getattr(llm_module, "chat")),
        "create_client": cast(
            CreateClient, getattr(llm_module, "create_llm_client")
        ),
        "get_extra_body": cast(
            GetExtraBody, getattr(llm_module, "get_extra_body")
        ),
        "request_error_types": (api_error_type,),
    }


def load_json(path: Path) -> JsonValue:
    """Load one UTF-8 JSON file with actionable path context."""

    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: path={path}")
    try:
        raw_value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise DownstreamDataError(
            "JSON file is malformed: "
            f"path={path}, line={error.lineno}, column={error.colno}, "
            f"message={error.msg}"
        ) from error
    return cast(JsonValue, json.loads(canonical_json(raw_value)))


def require_json_object(value: JsonValue, path: Path) -> dict[str, JsonValue]:
    """Return a JSON object or raise with file context."""

    if not isinstance(value, dict):
        raise DownstreamDataError(
            "JSON file must contain an object: "
            f"path={path}, value_type={type(value).__name__}"
        )
    return value


def require_json_list(value: JsonValue, path: Path) -> list[JsonValue]:
    """Return a JSON list or raise with file context."""

    if not isinstance(value, list):
        raise DownstreamDataError(
            "JSON file must contain a list: "
            f"path={path}, value_type={type(value).__name__}"
        )
    return value


def object_rows(values: Sequence[JsonValue], path: Path) -> list[JsonObject]:
    """Validate that every row is a JSON object."""

    rows: list[JsonObject] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise DownstreamDataError(
                "JSON row must be an object: "
                f"path={path}, index={index}, value_type={type(value).__name__}"
            )
        rows.append(cast(JsonObject, value))
    return rows


def indexed_rows(
    rows: Sequence[JsonObject],
    path: Path,
) -> dict[str, JsonObject]:
    """Index unique rows by instance_id."""

    output: dict[str, JsonObject] = {}
    for index, row in enumerate(rows):
        instance_id: JsonValue | None = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise DownstreamDataError(
                "Row has invalid instance_id: "
                f"path={path}, index={index}, value={instance_id!r}"
            )
        if instance_id in output:
            raise DownstreamDataError(
                f"Duplicate instance_id: path={path}, instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def load_corpus(path: Path) -> dict[str, JsonObject]:
    """Load a unique skill corpus indexed by skill_id."""

    rows: list[JsonObject] = object_rows(
        require_json_list(load_json(path), path),
        path,
    )
    output: dict[str, JsonObject] = {}
    for index, row in enumerate(rows):
        skill_id: JsonValue | None = row.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id:
            raise DownstreamDataError(
                "Corpus row has invalid skill_id: "
                f"path={path}, index={index}, value={skill_id!r}"
            )
        if skill_id in output:
            raise DownstreamDataError(
                f"Corpus contains duplicate skill_id: path={path}, skill_id={skill_id}"
            )
        output[skill_id] = row
    return output


def selector_generation(extra_body: JsonLike) -> SelectorGeneration:
    """Return the frozen selector generation identity."""

    normalized_extra_body: JsonValue = cast(
        JsonValue, json.loads(canonical_json(extra_body))
    )
    return {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "extra_body": normalized_extra_body,
        "max_parse_attempts": MAX_PARSE_ATTEMPTS,
        "rank1_fallback": True,
    }


def candidate_displays(
    candidates: Sequence[JsonObject],
    display_name: DisplayName,
) -> list[CandidateDisplay]:
    """Build the exact ordered candidate identity shown to the model."""

    output: list[CandidateDisplay] = []
    for index, skill in enumerate(candidates, start=1):
        skill_id: JsonValue | None = skill.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id:
            raise DownstreamDataError(
                f"Candidate has invalid skill_id: rank={index}, value={skill_id!r}"
            )
        raw_description: JsonValue | None = skill.get("description", "")
        description: str = (
            raw_description if isinstance(raw_description, str) else str(raw_description)
        )
        output.append(
            {
                "skill_id": skill_id,
                "name": display_name(skill, index),
                "description": description,
            }
        )
    return output


def build_selector_inputs(
    instances: Sequence[JsonObject],
    source_records: Mapping[str, JsonObject],
    corpus: Mapping[str, JsonObject],
    runtime: SelectorRuntime,
    manifest: RuntimeManifest,
    code_bundle_sha256_value: str,
) -> list[SelectorInput]:
    """Validate top-50 inputs and render every native selector request."""

    inputs: list[SelectorInput] = []
    for instance in instances:
        instance_id: JsonValue | None = instance.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise DownstreamDataError(
                f"Instance has invalid instance_id: value={instance_id!r}"
            )
        if instance_id not in source_records:
            raise DownstreamDataError(
                f"Retrieval source is missing instance: instance_id={instance_id}"
            )
        source_record: JsonObject = source_records[instance_id]
        raw_retrieved: JsonValue | None = source_record.get("retrieved")
        if not isinstance(raw_retrieved, list):
            raise DownstreamDataError(
                "Retrieval record has invalid retrieved field: "
                f"instance_id={instance_id}, value_type={type(raw_retrieved).__name__}"
            )
        if len(raw_retrieved) < POOL_SIZE:
            raise DownstreamDataError(
                "Selection source does not contain the frozen top-50 pool: "
                f"instance_id={instance_id}, retrieved={len(raw_retrieved)}, "
                f"required={POOL_SIZE}"
            )
        retrieved_rows: list[JsonObject] = []
        for rank, raw_row in enumerate(raw_retrieved[:POOL_SIZE], start=1):
            if not isinstance(raw_row, dict):
                raise DownstreamDataError(
                    "Retrieved candidate must be an object: "
                    f"instance_id={instance_id}, rank={rank}, "
                    f"value_type={type(raw_row).__name__}"
                )
            retrieved_rows.append(cast(JsonObject, raw_row))
        candidate_ids: list[str] = []
        candidates: list[JsonObject] = []
        for rank, retrieved in enumerate(retrieved_rows, start=1):
            skill_id: JsonValue | None = retrieved.get("skill_id")
            if not isinstance(skill_id, str) or not skill_id:
                raise DownstreamDataError(
                    "Retrieved candidate has invalid skill_id: "
                    f"instance_id={instance_id}, rank={rank}, value={skill_id!r}"
                )
            if skill_id not in corpus:
                raise DownstreamDataError(
                    "Retrieved candidate is absent from corpus: "
                    f"instance_id={instance_id}, rank={rank}, skill_id={skill_id}"
                )
            candidate_ids.append(skill_id)
            candidates.append(corpus[skill_id])
        if len(candidate_ids) != len(set(candidate_ids)):
            raise DownstreamDataError(
                "Selector top-50 contains duplicate skill IDs: "
                f"instance_id={instance_id}"
            )
        _, query = runtime["build_prompt"](instance, None)
        formatted_candidates: str = runtime["format_candidates"](candidates)
        rendered_prompt: str = runtime["prompt_template"].format(
            query=query,
            candidates=formatted_candidates,
        )
        displays: list[CandidateDisplay] = candidate_displays(
            candidates,
            runtime["display_name"],
        )
        generation: SelectorGeneration = selector_generation(
            runtime["get_extra_body"](
                cast(str, manifest["runtime_identity"].get("served_model", "")),
                THINKING,
            )
        )
        request_hash: str = selector_request_fingerprint(
            SELECTOR_SCHEMA_VERSION,
            instance_id,
            instance,
            rendered_prompt,
            displays,
            manifest["corpus_sha256"],
            manifest["runtime_identity"],
            generation,
            code_bundle_sha256_value,
        )
        inputs.append(
            {
                "instance": instance,
                "source_record": source_record,
                "candidates": candidates,
                "candidate_displays": displays,
                "rendered_prompt": rendered_prompt,
                "candidate_hash": sha256_json(displays),
                "request_hash": request_hash,
            }
        )
    return inputs


def error_context(error: Exception) -> tuple[str, int | None, str]:
    """Extract explicit exception, HTTP status, and response context."""

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
        response_body: str = ""
    elif isinstance(raw_body, str):
        response_body = raw_body
    else:
        try:
            response_body = canonical_json(raw_body)
        except DownstreamDataError:
            response_body = repr(raw_body)
    return exception_name, status_code, response_body


def append_attempt(
    path: Path,
    payload: Mapping[str, JsonLike],
    write_lock: threading.Lock,
) -> None:
    """Append one structured request attempt to the audit log."""

    line: str = canonical_json(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with write_lock:
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(line)
            output_file.flush()


def request_selector_response(
    selector_input: SelectorInput,
    client: OpenAI,
    model: str,
    extra_body: JsonObject | None,
    runtime: SelectorRuntime,
    parse_attempt: int,
    attempt_log: Path,
    initial_client_calls: int,
    write_lock: threading.Lock,
) -> tuple[str, int]:
    """Run one parse attempt with bounded infrastructure retries."""

    client_calls: int = initial_client_calls
    instance_id: str = cast(str, selector_input["instance"]["instance_id"])
    for infra_attempt in range(1, MAX_INFRA_ATTEMPTS + 1):
        client_calls += 1
        started_at: float = time.time()
        try:
            response: str = runtime["chat"](
                client,
                model,
                selector_input["rendered_prompt"],
                system=None,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stop=None,
                extra_body=extra_body,
            )
            append_attempt(
                attempt_log,
                {
                    "instance_id": instance_id,
                    "selector_request_hash": selector_input["request_hash"],
                    "parse_attempt": parse_attempt,
                    "infra_attempt": infra_attempt,
                    "client_call_attempt": client_calls,
                    "status": "response",
                    "elapsed_seconds": round(time.time() - started_at, 6),
                    "raw_response": response,
                },
                write_lock,
            )
            return response, client_calls
        except runtime["request_error_types"] as error:
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
                    "selector_request_hash": selector_input["request_hash"],
                    "parse_attempt": parse_attempt,
                    "infra_attempt": infra_attempt,
                    "client_call_attempt": client_calls,
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
                and infra_attempt < MAX_INFRA_ATTEMPTS
            )
            if not should_retry:
                raise SelectionCallFailure(
                    category,
                    exception_name,
                    str(error),
                    status_code,
                    response_body,
                    client_calls,
                ) from error
            warning: JsonObject = {
                "level": "warning",
                "event": "selector_infra_retry",
                "model": model,
                "instance_id": instance_id,
                "parse_attempt": parse_attempt,
                "infra_attempt": infra_attempt,
                "status_code": status_code,
                "response_body": response_body,
                "exception_name": exception_name,
            }
            print(canonical_json(warning), file=sys.stderr, flush=True)
            time.sleep(RETRY_DELAYS_SECONDS[infra_attempt - 1])
    raise AssertionError("Infrastructure retry loop exited without a result")


def select_one(
    selector_input: SelectorInput,
    client: OpenAI,
    model: str,
    extra_body: JsonObject | None,
    runtime: SelectorRuntime,
    attempt_log: Path,
    write_lock: threading.Lock,
) -> SelectionOutcome:
    """Apply the native three-response parse and rank-1 fallback protocol."""

    raw_responses: list[str] = []
    client_calls: int = 0
    candidate_count: int = len(selector_input["candidates"])
    for parse_attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
        try:
            response, client_calls = request_selector_response(
                selector_input,
                client,
                model,
                extra_body,
                runtime,
                parse_attempt,
                attempt_log,
                client_calls,
                write_lock,
            )
        except SelectionCallFailure as error:
            return {
                "selected_rank_zero_based": None,
                "raw_responses": raw_responses,
                "parse_attempts": len(raw_responses),
                "client_call_attempts": error.client_call_attempts,
                "parse_success": False,
                "rank1_fallback": False,
                "failure_category": error.category,
                "error": {
                    "exception_name": error.exception_name,
                    "message": str(error),
                    "status_code": error.status_code,
                    "response_body": error.response_body,
                },
            }
        raw_responses.append(response)
        selected_index: int | None = runtime["parse_first_number"](
            response,
            candidate_count,
        )
        if selected_index is not None:
            return {
                "selected_rank_zero_based": selected_index,
                "raw_responses": raw_responses,
                "parse_attempts": parse_attempt,
                "client_call_attempts": client_calls,
                "parse_success": True,
                "rank1_fallback": False,
                "failure_category": "success",
                "error": None,
            }
    return {
        "selected_rank_zero_based": 0,
        "raw_responses": raw_responses,
        "parse_attempts": MAX_PARSE_ATTEMPTS,
        "client_call_attempts": client_calls,
        "parse_success": False,
        "rank1_fallback": True,
        "failure_category": "selector_fallback",
        "error": None,
    }


def build_selection_record(
    selector_input: SelectorInput,
    outcome: SelectionOutcome,
    model: str,
    manifest: RuntimeManifest,
    code_bundle_sha256_value: str,
    source_sha256: str,
) -> SelectionRecord:
    """Attach immutable request provenance to one selector outcome."""

    instance_id: str = cast(str, selector_input["instance"]["instance_id"])
    dataset: JsonValue | None = selector_input["instance"].get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise DownstreamDataError(
            f"Instance has invalid dataset: instance_id={instance_id}, value={dataset!r}"
        )
    selected_index: int | None = outcome["selected_rank_zero_based"]
    selected_skill_id: str | None = (
        selector_input["candidate_displays"][selected_index]["skill_id"]
        if selected_index is not None
        else None
    )
    raw_responses: list[str] = outcome["raw_responses"]
    return {
        "schema_version": SELECTION_RECORD_SCHEMA_VERSION,
        "instance_id": instance_id,
        "dataset": dataset,
        "arm": "routed_select",
        "model": model,
        "ordered_candidate_ids": [
            candidate["skill_id"]
            for candidate in selector_input["candidate_displays"]
        ],
        "candidate_hash": selector_input["candidate_hash"],
        "selector_request_hash": selector_input["request_hash"],
        "selected_skill_id": selected_skill_id,
        "selected_rank": selected_index + 1 if selected_index is not None else None,
        "raw_response": raw_responses[-1] if raw_responses else "",
        "raw_responses": raw_responses,
        "parse_attempts": outcome["parse_attempts"],
        "client_call_attempts": outcome["client_call_attempts"],
        "parse_success": outcome["parse_success"],
        "rank1_fallback": outcome["rank1_fallback"],
        "failure_category": outcome["failure_category"],
        "runtime_identity": manifest["runtime_identity"],
        "code_bundle_sha256": code_bundle_sha256_value,
        "source_sha256": source_sha256,
        "error": outcome["error"],
    }


def read_selection_records(path: Path) -> list[SelectionRecord]:
    """Read and validate existing selection records for safe resume."""

    if not path.exists():
        return []
    records: list[SelectionRecord] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_record: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise DownstreamDataError(
                    "Selection JSONL is malformed: "
                    f"path={path}, line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            if not isinstance(raw_record, dict):
                raise DownstreamDataError(
                    "Selection JSONL row must be an object: "
                    f"path={path}, line={line_number}"
                )
            instance_id: JsonValue | None = raw_record.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise DownstreamDataError(
                    "Selection record has invalid instance_id: "
                    f"path={path}, line={line_number}, value={instance_id!r}"
                )
            validate_failure_category(raw_record.get("failure_category"))
            records.append(cast(SelectionRecord, raw_record))
    record_ids: list[str] = [record["instance_id"] for record in records]
    coverage = audit_record_coverage(
        list(dict.fromkeys(record_ids)),
        record_ids,
    )
    if coverage.duplicate_ids:
        raise DownstreamDataError(
            "Selection output contains duplicate records: "
            f"path={path}, duplicate_ids={list(coverage.duplicate_ids)[:20]}"
        )
    return records


def rewrite_without_transient_records(
    path: Path,
    records: Sequence[SelectionRecord],
) -> list[SelectionRecord]:
    """Atomically remove only retryable infrastructure failures."""

    retained: list[SelectionRecord] = [
        record
        for record in records
        if record["failure_category"] != "infra_transient"
    ]
    if len(retained) == len(records):
        return retained
    write_jsonl_atomic(path, retained)
    return retained


def write_jsonl_atomic(
    path: Path,
    records: Sequence[Mapping[str, JsonLike]],
) -> None:
    """Atomically replace one JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(canonical_json(record) + "\n")
    temporary_path.replace(path)


def append_selection_record(
    path: Path,
    record: SelectionRecord,
    write_lock: threading.Lock,
) -> None:
    """Append and flush one complete selection record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line: str = canonical_json(record) + "\n"
    with write_lock:
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(line)
            output_file.flush()


def build_selected_source(
    source_payload: Mapping[str, JsonLike],
    source_records: Sequence[JsonObject],
    selection_records: Mapping[str, SelectionRecord],
    source_sha256: str,
    selection_output_sha256: str,
) -> JsonObject:
    """Build a standard top-1 retrieval file from persisted decisions."""

    output_results: list[JsonObject] = []
    unresolved: int = 0
    method_failures: int = 0
    for source_record in source_records:
        instance_id: str = cast(str, source_record["instance_id"])
        selection_record: SelectionRecord | None = selection_records.get(instance_id)
        selected_retrieved: list[JsonObject] = []
        if selection_record is None:
            unresolved += 1
        elif selection_record["failure_category"] in ("success", "selector_fallback"):
            selected_skill_id: str | None = selection_record["selected_skill_id"]
            raw_retrieved: JsonValue | None = source_record.get("retrieved")
            if not isinstance(raw_retrieved, list):
                raise DownstreamDataError(
                    "Source retrieval field changed after input validation: "
                    f"instance_id={instance_id}"
                )
            matches: list[JsonObject] = [
                cast(JsonObject, row)
                for row in raw_retrieved[:POOL_SIZE]
                if isinstance(row, dict) and row.get("skill_id") == selected_skill_id
            ]
            if len(matches) != 1:
                raise DownstreamDataError(
                    "Selected skill does not map to exactly one source candidate: "
                    f"instance_id={instance_id}, selected_skill_id={selected_skill_id}, "
                    f"matches={len(matches)}"
                )
            selected_retrieved = [dict(matches[0])]
        elif selection_record["failure_category"] == "method_failure":
            method_failures += 1
        else:
            unresolved += 1
        output_results.append(
            {
                "instance_id": instance_id,
                "gold_skill_ids": list(
                    cast(Sequence[JsonValue], source_record.get("gold_skill_ids", []))
                ),
                "retrieved": selected_retrieved,
            }
        )
    raw_metadata: JsonLike = source_payload.get("metadata", {})
    metadata: JsonObject = (
        dict(cast(Mapping[str, JsonLike], raw_metadata))
        if isinstance(raw_metadata, dict)
        else {}
    )
    metadata["selector"] = {
        "schema_version": SELECTION_RECORD_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "selection_output_sha256": selection_output_sha256,
        "pool": POOL_SIZE,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "max_parse_attempts": MAX_PARSE_ATTEMPTS,
        "rank1_fallback": True,
        "records": len(selection_records),
        "unresolved": unresolved,
        "method_failures": method_failures,
        "complete": unresolved == 0 and len(selection_records) == len(source_records),
    }
    return {
        "metadata": metadata,
        "results": output_results,
    }


def write_json_atomic(path: Path, payload: Mapping[str, JsonLike]) -> None:
    """Atomically replace one formatted JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    """Run one resumable selection-only model-domain job."""

    args = parse_args()
    workers: int = int(args.workers)
    if workers <= 0:
        raise ValueError(f"workers must be positive: workers={workers}")
    runtime: SelectorRuntime = load_selector_runtime()
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    source_path: Path = cast(Path, args.source).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    selected_source_path: Path = cast(Path, args.selected_source).resolve()
    attempt_log_path: Path = cast(Path, args.attempt_log).resolve()
    manifest_path: Path = cast(Path, args.runtime_manifest).resolve()
    model: str = str(args.model)
    api_base: str = str(args.api_base)
    domain: str = str(args.domain)
    code_bundle_sha256_value: str = str(args.code_bundle_sha256)

    instance_rows: list[JsonObject] = object_rows(
        require_json_list(load_json(instances_path), instances_path),
        instances_path,
    )
    for instance in instance_rows:
        if instance.get("dataset") != domain:
            raise DownstreamDataError(
                "Instance domain mismatch: "
                f"instance_id={instance.get('instance_id')}, "
                f"expected={domain}, actual={instance.get('dataset')}"
            )
    instance_index: dict[str, JsonObject] = indexed_rows(
        instance_rows,
        instances_path,
    )
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    source_payload_json: dict[str, JsonValue] = require_json_object(
        load_json(source_path),
        source_path,
    )
    raw_source_results: JsonValue = source_payload_json.get("results")
    if not isinstance(raw_source_results, list):
        raise DownstreamDataError(
            f"Retrieval source must contain results list: path={source_path}"
        )
    source_rows: list[JsonObject] = object_rows(
        raw_source_results,
        source_path,
    )
    source_index: dict[str, JsonObject] = indexed_rows(
        source_rows,
        source_path,
    )
    source_coverage = audit_record_coverage(
        list(instance_index),
        list(source_index),
    )
    if not source_coverage.complete:
        raise DownstreamDataError(
            "Retrieval source coverage mismatch: "
            f"missing={list(source_coverage.missing_ids)[:20]}, "
            f"unexpected={list(source_coverage.unexpected_ids)[:20]}"
        )
    manifest_json: JsonValue = load_json(manifest_path)
    manifest: RuntimeManifest = validate_selector_runtime_manifest(
        manifest_json,
        sha256_file(instances_path),
        sha256_file(corpus_path),
        code_bundle_sha256_value,
    )
    served_model: JsonValue | None = manifest["runtime_identity"].get("served_model")
    if served_model != model:
        raise DownstreamDataError(
            "Selector model does not match runtime manifest: "
            f"cli_model={model}, manifest_served_model={served_model!r}"
        )
    selector_inputs: list[SelectorInput] = build_selector_inputs(
        instance_rows,
        source_index,
        corpus,
        runtime,
        manifest,
        code_bundle_sha256_value,
    )
    existing_records: list[SelectionRecord] = rewrite_without_transient_records(
        output_path,
        read_selection_records(output_path),
    )
    existing_index: dict[str, SelectionRecord] = {
        record["instance_id"]: record for record in existing_records
    }
    pending_inputs: list[SelectorInput] = [
        selector_input
        for selector_input in selector_inputs
        if cast(str, selector_input["instance"]["instance_id"]) not in existing_index
    ]
    client: OpenAI = runtime["create_client"](api_base, None)
    extra_body: JsonObject | None = runtime["get_extra_body"](
        model,
        THINKING,
    )
    source_sha256: str = sha256_file(source_path)
    write_lock: threading.Lock = threading.Lock()

    def run_one(selector_input: SelectorInput) -> SelectionRecord:
        outcome: SelectionOutcome = select_one(
            selector_input,
            client,
            model,
            extra_body,
            runtime,
            attempt_log_path,
            write_lock,
        )
        return build_selection_record(
            selector_input,
            outcome,
            model,
            manifest,
            code_bundle_sha256_value,
            source_sha256,
        )

    futures: dict[Future[SelectionRecord], str] = {}
    if pending_inputs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for selector_input in pending_inputs:
                instance_id: str = cast(
                    str, selector_input["instance"]["instance_id"]
                )
                futures[executor.submit(run_one, selector_input)] = instance_id
            completed: int = 0
            for future in as_completed(futures):
                record: SelectionRecord = future.result()
                append_selection_record(output_path, record, write_lock)
                existing_index[record["instance_id"]] = record
                completed += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        canonical_json(
                            {
                                "event": "selection_progress",
                                "domain": domain,
                                "model": model,
                                "completed_this_run": completed,
                                "pending_this_run": len(futures),
                                "total_records": len(existing_index),
                            }
                        ),
                        flush=True,
                    )

    final_records: list[SelectionRecord] = read_selection_records(output_path)
    final_index: dict[str, SelectionRecord] = {
        record["instance_id"]: record for record in final_records
    }
    coverage = audit_record_coverage(
        list(instance_index),
        [record["instance_id"] for record in final_records],
    )
    selection_output_sha256: str = sha256_file(output_path)
    selected_source_payload: JsonObject = build_selected_source(
        cast(Mapping[str, JsonLike], source_payload_json),
        source_rows,
        final_index,
        source_sha256,
        selection_output_sha256,
    )
    write_json_atomic(selected_source_path, selected_source_payload)
    category_counts: dict[str, int] = {}
    for record in final_records:
        category: str = record["failure_category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    summary: JsonObject = {
        "event": "selection_complete",
        "model": model,
        "domain": domain,
        "expected": len(instance_rows),
        "observed": len(final_records),
        "missing": len(coverage.missing_ids),
        "duplicates": len(coverage.duplicate_ids),
        "unexpected": len(coverage.unexpected_ids),
        "failure_categories": category_counts,
        "output": str(output_path),
        "selected_source": str(selected_source_path),
    }
    print(canonical_json(summary), flush=True)
    unresolved: int = (
        category_counts.get("infra_transient", 0)
        + category_counts.get("unclassified_error", 0)
    )
    if not coverage.complete or unresolved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
