"""Shared execution evidence for runtime-matched baseline jobs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Literal, NamedTuple, Protocol, TypeAlias, TypedDict, cast


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonLike: TypeAlias = (
    JsonScalar | Mapping[str, "JsonLike"] | Sequence["JsonLike"]
)
FailureCategory: TypeAlias = Literal[
    "success",
    "selector_fallback",
    "infra_transient",
    "method_failure",
    "unclassified_error",
]

SHA256_PATTERN: re.Pattern[str] = re.compile(r"[0-9a-f]{64}")
RUNTIME_FACTS_SCHEMA_VERSION: str = "runtime-matched-runtime-facts-v1"
JOB_MANIFEST_SCHEMA_VERSION: str = "runtime-matched-job-manifest-v1"


class RuntimeMatchedExecutionError(ValueError):
    """Raised when execution evidence violates the frozen protocol."""


class UsageAttributionError(RuntimeError):
    """Raised when one HTTP call cannot be bound to a logical attempt."""


class RuntimeManifestError(RuntimeMatchedExecutionError):
    """Raised when a job-bound runtime manifest is incomplete or inconsistent."""


class ExecutionContext(NamedTuple):
    """Immutable attribution for one logical engine or decision attempt."""

    job_id: str
    model: str
    domain: str
    arm: str
    instance_id: str
    logical_attempt: int
    answer_payload_hash: str
    execution_request_hash: str


class ErrorContext(NamedTuple):
    """Structured context extracted from one request exception."""

    exception_name: str
    message: str
    status_code: int | None
    response_body: str


class TokenUsage(TypedDict):
    """Token usage returned by one successful OpenAI-compatible response."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    usage_missing_reason: str | None


class FileEvidence(TypedDict):
    """One named immutable file input."""

    name: str
    path: str
    size_bytes: int
    sha256: str


class CodeFileEvidence(TypedDict):
    """One repository-relative source file."""

    path: str
    size_bytes: int
    sha256: str


class JobBoundManifest(TypedDict):
    """Validated runtime, input, generation, and code identity for one job."""

    schema_version: str
    runtime_facts: dict[str, JsonValue]
    generation: dict[str, JsonValue]
    artifacts: list[FileEvidence]
    code_files: list[CodeFileEvidence]
    code_bundle_sha256: str


class FrozenK2RuntimeReference(TypedDict):
    """Authoritative K=2 endpoint identity for one result tag."""

    served_model: str
    checkpoint_repository: str
    checkpoint_revision: str
    checkpoint_files_manifest_sha256: str
    tokenizer_artifacts: dict[str, str]
    chat_template_sha256: str
    vllm_version: str


class UsageSink(Protocol):
    """Persistence callback for one attributed HTTP usage event."""

    def __call__(self, event: Mapping[str, JsonLike]) -> None:
        """Persist one complete event or raise."""


class ChatCompletionsLike(Protocol):
    """OpenAI-compatible chat-completions resource."""

    def create(self, *args: object, **kwargs: object) -> object:
        """Submit one chat-completions request."""


class ChatLike(Protocol):
    """OpenAI-compatible chat resource."""

    completions: ChatCompletionsLike


class OpenAIClientLike(Protocol):
    """Minimum client surface used by native SR-Agents components."""

    chat: ChatLike


_JOB_ID: ContextVar[str | None] = ContextVar("runtime_job_id", default=None)
_MODEL: ContextVar[str | None] = ContextVar("runtime_model", default=None)
_DOMAIN: ContextVar[str | None] = ContextVar("runtime_domain", default=None)
_ARM: ContextVar[str | None] = ContextVar("runtime_arm", default=None)
_INSTANCE_ID: ContextVar[str | None] = ContextVar(
    "runtime_instance_id",
    default=None,
)
_LOGICAL_ATTEMPT: ContextVar[int | None] = ContextVar(
    "runtime_logical_attempt",
    default=None,
)
_HTTP_SUBCALL: ContextVar[int | None] = ContextVar(
    "runtime_http_subcall",
    default=None,
)
_ANSWER_PAYLOAD_HASH: ContextVar[str | None] = ContextVar(
    "runtime_answer_payload_hash",
    default=None,
)
_EXECUTION_REQUEST_HASH: ContextVar[str | None] = ContextVar(
    "runtime_execution_request_hash",
    default=None,
)

FROZEN_SRAGENTS_REVISION: str = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
FROZEN_K2_RUNTIME_REFERENCES: dict[str, FrozenK2RuntimeReference] = {
    "deepseek7b": {
        "served_model": "deepseek7b",
        "checkpoint_repository": "deepseek-ai/deepseek-llm-7b-chat",
        "checkpoint_revision": "snapshots/master",
        "checkpoint_files_manifest_sha256": (
            "25b7f08040a12a38ed6a4fdca625063e18091926a30813d56a3c87e3cbe1f03b"
        ),
        "tokenizer_artifacts": {
            "tokenizer.json": (
                "a08b02921f08548065a7b2ec13b2ffeed873231add60f9c3c7b08b04f2cc212a"
            ),
            "tokenizer_config.json": (
                "9e4d4a34afe6db6096508a5363b065cf684ec3a9047da1c2dbe30bd8537a6086"
            ),
        },
        "chat_template_sha256": (
            "9e4d4a34afe6db6096508a5363b065cf684ec3a9047da1c2dbe30bd8537a6086"
        ),
        "vllm_version": "0.19.1",
    },
    "glm4-9b": {
        "served_model": "glm4-9b",
        "checkpoint_repository": "ZhipuAI/glm-4-9b-chat",
        "checkpoint_revision": "snapshots/master",
        "checkpoint_files_manifest_sha256": (
            "cd37e55587031d4dbc51bf768f83268669e196434f209b1bd0e6245991e038be"
        ),
        "tokenizer_artifacts": {
            "tokenizer.model": (
                "5a493598071550244b2ee7f26118f3edec2150b9dfa967929a99052ac83fe716"
            ),
            "tokenizer_config.json": (
                "f891e4d4ebb4009b6996dea97befb77a60c0cef0e88ac1edd6c741b1367f9c62"
            ),
        },
        "chat_template_sha256": (
            "f891e4d4ebb4009b6996dea97befb77a60c0cef0e88ac1edd6c741b1367f9c62"
        ),
        "vllm_version": "0.19.1",
    },
    "llama31-8b": {
        "served_model": "llama31-8b",
        "checkpoint_repository": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "checkpoint_revision": "snapshots/master",
        "checkpoint_files_manifest_sha256": (
            "a8e51a9052d5cfe3faea783aa90837c6ba39d04f438eb6eca344a0f4b1e44630"
        ),
        "tokenizer_artifacts": {
            "tokenizer.json": (
                "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"
            ),
            "tokenizer_config.json": (
                "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424"
            ),
        },
        "chat_template_sha256": (
            "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424"
        ),
        "vllm_version": "0.19.1",
    },
    "mistral7b": {
        "served_model": "mistral7b",
        "checkpoint_repository": "LLM-Research/Mistral-7B-Instruct-v0.3",
        "checkpoint_revision": "c8cfccbcfd71d4e3479498c30b2823bab19c4687",
        "checkpoint_files_manifest_sha256": (
            "559840283ece7b8cbbb937d74d5ce47aff520cda4a453a3331ac3e8f26bfa6df"
        ),
        "tokenizer_artifacts": {
            "tokenizer.json": (
                "60b945759e27a63c3c5c0ca675881f5a73b4aa38b5d1d6818570308d4f1a3c59"
            ),
            "tokenizer.model": (
                "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89"
            ),
            "tokenizer_config.json": (
                "b0c776216a54c6d031866d1dff0b31715bd73f5ba87f8a30eb35e8c603dff95d"
            ),
        },
        "chat_template_sha256": (
            "b0c776216a54c6d031866d1dff0b31715bd73f5ba87f8a30eb35e8c603dff95d"
        ),
        "vllm_version": "0.19.1",
    },
    "qwen3.5-4b-reference": {
        "served_model": "qwen3.5-4b",
        "checkpoint_repository": "Qwen/Qwen3.5-4B",
        "checkpoint_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "checkpoint_files_manifest_sha256": (
            "7447e4e49652e2eb494c53d808d9b4e005838b1430aecb6df8181b2105d177dc"
        ),
        "tokenizer_artifacts": {
            "tokenizer_config.json": (
                "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
            ),
        },
        "chat_template_sha256": (
            "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
        ),
        "vllm_version": "0.17.1",
    },
    "qwen35-9b": {
        "served_model": "qwen35-9b",
        "checkpoint_repository": "Qwen/Qwen3.5-9B",
        "checkpoint_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "checkpoint_files_manifest_sha256": (
            "daf8a250ee437249688f839397f7908ed75e10eba31ab9a5663456c36c46b595"
        ),
        "tokenizer_artifacts": {
            "tokenizer.json": (
                "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
            ),
            "tokenizer_config.json": (
                "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
            ),
        },
        "chat_template_sha256": (
            "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
        ),
        "vllm_version": "0.17.1",
    },
    "yi15-9b": {
        "served_model": "yi15-9b",
        "checkpoint_repository": "01ai/Yi-1.5-9B-Chat",
        "checkpoint_revision": "snapshots/master",
        "checkpoint_files_manifest_sha256": (
            "45eb2167b36e6209f26a897a440cf27bf002f4b1368556d9105fbe76341addca"
        ),
        "tokenizer_artifacts": {
            "tokenizer.json": (
                "a13ccc285aea27f5e9a98d40e04e330b01d89db6de7af10b013f56eec8eae8a2"
            ),
            "tokenizer.model": (
                "386c49cf943d71aa110361135338c50e38beeff0a66593480421f37b319e1a39"
            ),
            "tokenizer_config.json": (
                "a877a66153e25d07e7ac73fa33f4d4003cb8bdd93bab1a32fc0b4578554ccba4"
            ),
        },
        "chat_template_sha256": (
            "a877a66153e25d07e7ac73fa33f4d4003cb8bdd93bab1a32fc0b4578554ccba4"
        ),
        "vllm_version": "0.19.1",
    },
}


def _json_value(value: JsonLike, location: str) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeMatchedExecutionError(
                f"Non-finite JSON number: location={location}, value={value}"
            )
        return value
    if isinstance(value, Mapping):
        output: dict[str, JsonValue] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise RuntimeMatchedExecutionError(
                    "JSON object key must be a string: "
                    f"location={location}, key_type={type(raw_key).__name__}"
                )
            output[raw_key] = _json_value(
                raw_value,
                f"{location}.{raw_key}",
            )
        return output
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _json_value(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RuntimeMatchedExecutionError(
        "Unsupported canonical JSON value: "
        f"location={location}, value_type={type(value).__name__}"
    )


def canonical_json(value: JsonLike) -> str:
    """Serialize one JSON-compatible value using the frozen canonical form."""

    return json.dumps(
        _json_value(value, "$"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 digest of UTF-8 text."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: JsonLike) -> str:
    """Return the SHA-256 digest of canonical JSON."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    if not path.is_file():
        raise FileNotFoundError(f"Hash input is not a regular file: path={path}")
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: JsonLike | None, field_name: str) -> str:
    """Return one lowercase SHA-256 digest or raise."""

    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeMatchedExecutionError(
            "Field must be a lowercase SHA-256 digest: "
            f"field={field_name}, value={value!r}"
        )
    return value


def require_nonempty_string(value: JsonLike | None, field_name: str) -> str:
    """Return one non-empty string or raise."""

    if not isinstance(value, str) or not value.strip():
        raise RuntimeMatchedExecutionError(
            f"Field must be a non-empty string: field={field_name}, value={value!r}"
        )
    return value


def require_positive_integer(value: JsonLike | None, field_name: str) -> int:
    """Return one positive integer or raise."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeMatchedExecutionError(
            f"Field must be a positive integer: field={field_name}, value={value!r}"
        )
    return value


def require_json_object(
    value: JsonLike | None,
    field_name: str,
) -> dict[str, JsonValue]:
    """Return one normalized JSON object or raise."""

    normalized: JsonValue = _json_value(value, field_name)
    if not isinstance(normalized, dict):
        raise RuntimeMatchedExecutionError(
            "Field must be a JSON object: "
            f"field={field_name}, value_type={type(normalized).__name__}"
        )
    return normalized


def load_json_file(path: Path, context: str) -> JsonValue:
    """Load one UTF-8 JSON file with path-aware errors."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    try:
        return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise RuntimeMatchedExecutionError(
            f"{context} JSON is malformed: path={path}, line={error.lineno}, "
            f"column={error.colno}, message={error.msg}"
        ) from error


def write_json_atomic(path: Path, payload: JsonLike) -> None:
    """Atomically write one formatted JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            _json_value(payload, "$"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _validate_context(context: ExecutionContext) -> None:
    for field_name, value in (
        ("job_id", context.job_id),
        ("model", context.model),
        ("domain", context.domain),
        ("arm", context.arm),
        ("instance_id", context.instance_id),
    ):
        require_nonempty_string(value, field_name)
    if context.logical_attempt <= 0:
        raise UsageAttributionError(
            "logical_attempt must be positive: "
            f"value={context.logical_attempt}, instance_id={context.instance_id}"
        )
    require_sha256(context.answer_payload_hash, "answer_payload_hash")
    require_sha256(context.execution_request_hash, "execution_request_hash")


@contextmanager
def bind_execution_context(context: ExecutionContext) -> Iterator[None]:
    """Bind one complete logical-attempt context in the current thread."""

    _validate_context(context)
    tokens: tuple[tuple[ContextVar[object], Token[object]], ...] = (
        cast(
            tuple[ContextVar[object], Token[object]],
            (_JOB_ID, _JOB_ID.set(context.job_id)),
        ),
        cast(
            tuple[ContextVar[object], Token[object]],
            (_MODEL, _MODEL.set(context.model)),
        ),
        cast(
            tuple[ContextVar[object], Token[object]],
            (_DOMAIN, _DOMAIN.set(context.domain)),
        ),
        cast(
            tuple[ContextVar[object], Token[object]],
            (_ARM, _ARM.set(context.arm)),
        ),
        cast(
            tuple[ContextVar[object], Token[object]],
            (_INSTANCE_ID, _INSTANCE_ID.set(context.instance_id)),
        ),
        cast(
            tuple[ContextVar[object], Token[object]],
            (_LOGICAL_ATTEMPT, _LOGICAL_ATTEMPT.set(context.logical_attempt)),
        ),
        cast(
            tuple[ContextVar[object], Token[object]],
            (_HTTP_SUBCALL, _HTTP_SUBCALL.set(0)),
        ),
        cast(
            tuple[ContextVar[object], Token[object]],
            (
                _ANSWER_PAYLOAD_HASH,
                _ANSWER_PAYLOAD_HASH.set(context.answer_payload_hash),
            ),
        ),
        cast(
            tuple[ContextVar[object], Token[object]],
            (
                _EXECUTION_REQUEST_HASH,
                _EXECUTION_REQUEST_HASH.set(context.execution_request_hash),
            ),
        ),
    )
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def _current_context() -> ExecutionContext:
    job_id: str | None = _JOB_ID.get()
    model: str | None = _MODEL.get()
    domain: str | None = _DOMAIN.get()
    arm: str | None = _ARM.get()
    instance_id: str | None = _INSTANCE_ID.get()
    logical_attempt: int | None = _LOGICAL_ATTEMPT.get()
    answer_hash: str | None = _ANSWER_PAYLOAD_HASH.get()
    execution_hash: str | None = _EXECUTION_REQUEST_HASH.get()
    missing: list[str] = [
        name
        for name, value in (
            ("job_id", job_id),
            ("model", model),
            ("domain", domain),
            ("arm", arm),
            ("instance_id", instance_id),
            ("logical_attempt", logical_attempt),
            ("answer_payload_hash", answer_hash),
            ("execution_request_hash", execution_hash),
        )
        if value is None
    ]
    if missing:
        raise UsageAttributionError(
            "HTTP request is outside a complete execution context: "
            f"missing={missing}"
        )
    return ExecutionContext(
        cast(str, job_id),
        cast(str, model),
        cast(str, domain),
        cast(str, arm),
        cast(str, instance_id),
        cast(int, logical_attempt),
        cast(str, answer_hash),
        cast(str, execution_hash),
    )


def _next_http_subcall() -> int:
    current: int | None = _HTTP_SUBCALL.get()
    if current is None:
        raise UsageAttributionError(
            "HTTP subcall counter is outside a bound execution context"
        )
    next_value: int = current + 1
    _HTTP_SUBCALL.set(next_value)
    return next_value


def _usage_field(usage: object, field_name: str) -> int | None:
    raw_value: object
    if isinstance(usage, Mapping):
        raw_value = usage.get(field_name)
    else:
        raw_value = getattr(usage, field_name, None)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        return None
    if raw_value < 0:
        return None
    return raw_value


def normalize_response_usage(response: object) -> TokenUsage:
    """Return actual response usage or explicit nulls with a reason."""

    usage: object | None = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "usage_missing_reason": "response_usage_absent",
        }
    prompt_tokens: int | None = _usage_field(usage, "prompt_tokens")
    completion_tokens: int | None = _usage_field(usage, "completion_tokens")
    total_tokens: int | None = _usage_field(usage, "total_tokens")
    missing_fields: list[str] = [
        field_name
        for field_name, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("total_tokens", total_tokens),
        )
        if value is None
    ]
    reason: str | None = None
    if missing_fields:
        reason = "response_usage_missing_or_invalid:" + ",".join(missing_fields)
    elif cast(int, prompt_tokens) + cast(int, completion_tokens) != total_tokens:
        reason = "response_usage_inconsistent_total"
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "usage_missing_reason": reason,
    }


def error_context(error: Exception) -> ErrorContext:
    """Extract exception type, status code, and response body."""

    exception_name: str = type(error).__name__
    raw_status: object = getattr(error, "status_code", None)
    status_code: int | None = (
        raw_status
        if isinstance(raw_status, int) and not isinstance(raw_status, bool)
        else None
    )
    raw_body: object = getattr(error, "body", None)
    if raw_body is None:
        raw_body = getattr(getattr(error, "response", None), "text", None)
    if raw_body is None:
        response_body: str = ""
    elif isinstance(raw_body, str):
        response_body = raw_body
    elif isinstance(raw_body, Mapping) or (
        isinstance(raw_body, Sequence)
        and not isinstance(raw_body, (str, bytes, bytearray))
    ):
        response_body = canonical_json(cast(JsonLike, raw_body))
    else:
        response_body = repr(raw_body)
    return ErrorContext(
        exception_name,
        str(error),
        status_code,
        response_body,
    )


def classify_request_error(
    exception_name: str,
    message: str,
    status_code: int | None,
    response_body: str,
) -> FailureCategory:
    """Classify one request failure without swallowing unknown failures."""

    combined: str = f"{message}\n{response_body}".lower()
    context_markers: tuple[str, ...] = (
        "context length",
        "context_length",
        "maximum context",
        "max_model_len",
        "too many tokens",
        "prompt is too long",
    )
    if status_code in (400, 413) and any(
        marker in combined for marker in context_markers
    ):
        return "method_failure"
    transient_names: frozenset[str] = frozenset(
        {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectionError",
            "InternalServerError",
            "RateLimitError",
            "ReadTimeout",
            "RemoteProtocolError",
            "Timeout",
            "TimeoutError",
        }
    )
    transient_statuses: frozenset[int] = frozenset(
        {408, 409, 425, 429, 500, 502, 503, 504}
    )
    if exception_name in transient_names or status_code in transient_statuses:
        return "infra_transient"
    return "unclassified_error"


def _usage_event_base(
    context: ExecutionContext,
    http_subcall: int,
    elapsed_seconds: float,
) -> dict[str, JsonLike]:
    return {
        "schema_version": "runtime-matched-usage-event-v1",
        "job_id": context.job_id,
        "model": context.model,
        "domain": context.domain,
        "arm": context.arm,
        "instance_id": context.instance_id,
        "logical_attempt": context.logical_attempt,
        "http_subcall": http_subcall,
        "answer_payload_hash": context.answer_payload_hash,
        "execution_request_hash": context.execution_request_hash,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


class _UsageCapturingCompletions:
    """Connector that records usage while preserving the original response."""

    def __init__(
        self,
        inner: ChatCompletionsLike,
        usage_sink: UsageSink,
    ) -> None:
        self._inner: ChatCompletionsLike = inner
        self._usage_sink: UsageSink = usage_sink

    def create(self, *args: object, **kwargs: object) -> object:
        context: ExecutionContext = _current_context()
        http_subcall: int = _next_http_subcall()
        started_at: float = time.monotonic()
        try:
            response: object = self._inner.create(*args, **kwargs)
        except Exception as error:
            details: ErrorContext = error_context(error)
            event: dict[str, JsonLike] = _usage_event_base(
                context,
                http_subcall,
                time.monotonic() - started_at,
            )
            event.update(
                {
                    "status": "error",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "usage_missing_reason": (
                        f"request_failed:{details.exception_name}"
                    ),
                    "exception_name": details.exception_name,
                    "message": details.message,
                    "status_code": details.status_code,
                    "response_body": details.response_body,
                }
            )
            self._usage_sink(event)
            raise
        usage: TokenUsage = normalize_response_usage(response)
        event = _usage_event_base(
            context,
            http_subcall,
            time.monotonic() - started_at,
        )
        event.update(
            {
                "status": "response",
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "usage_missing_reason": usage["usage_missing_reason"],
            }
        )
        self._usage_sink(event)
        return response

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


class _UsageCapturingChat:
    """Connector that replaces only the completions resource."""

    def __init__(self, inner: ChatLike, usage_sink: UsageSink) -> None:
        self._inner: ChatLike = inner
        self.completions: ChatCompletionsLike = _UsageCapturingCompletions(
            inner.completions,
            usage_sink,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


class _UsageCapturingClient:
    """Connector preserving a client while instrumenting chat completions."""

    def __init__(self, inner: OpenAIClientLike, usage_sink: UsageSink) -> None:
        self._inner: OpenAIClientLike = inner
        self.chat: ChatLike = _UsageCapturingChat(inner.chat, usage_sink)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def wrap_openai_client(
    client: OpenAIClientLike,
    usage_sink: UsageSink,
) -> OpenAIClientLike:
    """Wrap one client without changing any submitted request or response."""

    return _UsageCapturingClient(client, usage_sink)


def _loaded_skill_payload(
    loaded_skills: Sequence[Mapping[str, JsonLike]],
) -> list[dict[str, JsonLike]]:
    output: list[dict[str, JsonLike]] = []
    for index, skill in enumerate(loaded_skills):
        skill_id: str = require_nonempty_string(
            skill.get("skill_id"),
            f"loaded_skills[{index}].skill_id",
        )
        content: JsonLike | None = skill.get("content")
        if not isinstance(content, str):
            raise RuntimeMatchedExecutionError(
                "Loaded skill content must be a string: "
                f"index={index}, skill_id={skill_id}, "
                f"value_type={type(content).__name__}"
            )
        output.append({"skill_id": skill_id, "content": content})
    return output


def answer_payload_hash(
    schema_version: str,
    instance: Mapping[str, JsonLike],
    messages: Sequence[Mapping[str, JsonLike]],
    loaded_skills: Sequence[Mapping[str, JsonLike]],
    tools: Sequence[Mapping[str, JsonLike]],
    generation: Mapping[str, JsonLike],
) -> str:
    """Hash only the model-visible answer payload and generation parameters."""

    require_nonempty_string(schema_version, "schema_version")
    payload: dict[str, JsonLike] = {
        "schema_version": schema_version,
        "instance": instance,
        "messages": messages,
        "loaded_skills": _loaded_skill_payload(loaded_skills),
        "tools": tools,
        "generation": generation,
    }
    return sha256_json(payload)


def execution_request_hash(
    schema_version: str,
    answer_payload_sha256: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
) -> str:
    """Bind one answer payload to the exact runtime and executing code."""

    require_nonempty_string(schema_version, "schema_version")
    payload: dict[str, JsonLike] = {
        "schema_version": schema_version,
        "answer_payload_hash": require_sha256(
            answer_payload_sha256,
            "answer_payload_sha256",
        ),
        "runtime_manifest_sha256": require_sha256(
            runtime_manifest_sha256,
            "runtime_manifest_sha256",
        ),
        "code_bundle_sha256": require_sha256(
            code_bundle_sha256,
            "code_bundle_sha256",
        ),
    }
    return sha256_json(payload)


def _secret_locations(value: JsonValue, location: str) -> list[str]:
    if isinstance(value, dict):
        locations: list[str] = []
        for key, child in value.items():
            normalized_key: str = key.lower().replace("-", "_")
            if normalized_key in {
                "api_key",
                "apikey",
                "authorization",
                "password",
                "secret",
                "token",
                "access_token",
                "refresh_token",
            } or normalized_key.endswith(
                (
                    "_api_key",
                    "_password",
                    "_secret",
                    "_access_token",
                    "_refresh_token",
                )
            ):
                locations.append(f"{location}.{key}")
            locations.extend(_secret_locations(child, f"{location}.{key}"))
        return locations
    if isinstance(value, list):
        output: list[str] = []
        for index, child in enumerate(value):
            output.extend(_secret_locations(child, f"{location}[{index}]"))
        return output
    if isinstance(value, str) and re.search(
        r"(?i)(api[_-]?key|authorization|password|bearer\s+[a-z0-9])",
        value,
    ):
        return [location]
    return []


def _required_object(
    parent: Mapping[str, JsonValue],
    key: str,
    context: str,
) -> dict[str, JsonValue]:
    value: JsonValue | None = parent.get(key)
    if not isinstance(value, dict):
        raise RuntimeManifestError(
            f"Manifest field must be an object: field={context}.{key}"
        )
    return value


def _validate_runtime_facts(value: JsonLike) -> dict[str, JsonValue]:
    facts: dict[str, JsonValue] = require_json_object(value, "runtime_facts")
    if facts.get("schema_version") != RUNTIME_FACTS_SCHEMA_VERSION:
        raise RuntimeManifestError(
            "Runtime facts schema mismatch: "
            f"expected={RUNTIME_FACTS_SCHEMA_VERSION}, "
            f"actual={facts.get('schema_version')!r}"
        )
    secret_locations: list[str] = _secret_locations(facts, "$.runtime_facts")
    if secret_locations:
        raise RuntimeManifestError(
            "Runtime facts contain credential-like material: "
            f"locations={secret_locations}"
        )
    job: dict[str, JsonValue] = _required_object(facts, "job", "runtime_facts")
    for field_name in (
        "job_id",
        "result_tag",
        "model",
        "domain",
        "arm",
    ):
        require_nonempty_string(job.get(field_name), f"job.{field_name}")
    checkpoint: dict[str, JsonValue] = _required_object(
        facts,
        "checkpoint",
        "runtime_facts",
    )
    for field_name in (
        "repository",
        "revision",
        "path",
        "provenance",
    ):
        require_nonempty_string(
            checkpoint.get(field_name),
            f"checkpoint.{field_name}",
        )
    require_sha256(
        checkpoint.get("files_manifest_sha256"),
        "checkpoint.files_manifest_sha256",
    )
    tokenizer: dict[str, JsonValue] = _required_object(
        facts,
        "tokenizer",
        "runtime_facts",
    )
    tokenizer_artifacts: dict[str, JsonValue] = require_json_object(
        tokenizer.get("artifacts"),
        "tokenizer.artifacts",
    )
    if not tokenizer_artifacts:
        raise RuntimeManifestError("tokenizer.artifacts must not be empty")
    for artifact_name, digest in tokenizer_artifacts.items():
        require_sha256(digest, f"tokenizer.artifacts.{artifact_name}")
    require_sha256(
        tokenizer.get("chat_template_sha256"),
        "tokenizer.chat_template_sha256",
    )
    endpoint: dict[str, JsonValue] = _required_object(
        facts,
        "endpoint",
        "runtime_facts",
    )
    for field_name in (
        "api_base",
        "served_model",
        "process_command",
        "vllm_version",
        "dtype",
        "quantization",
    ):
        require_nonempty_string(endpoint.get(field_name), f"endpoint.{field_name}")
    require_positive_integer(
        endpoint.get("max_model_len"),
        "endpoint.max_model_len",
    )
    require_positive_integer(
        endpoint.get("tensor_parallel_size"),
        "endpoint.tensor_parallel_size",
    )
    if endpoint.get("served_model") != job.get("model"):
        raise RuntimeManifestError(
            "Job model and endpoint served model differ: "
            f"job={job.get('model')!r}, endpoint={endpoint.get('served_model')!r}"
        )
    models_readback: dict[str, JsonValue] = require_json_object(
        endpoint.get("models_readback"),
        "endpoint.models_readback",
    )
    raw_models: JsonValue | None = models_readback.get("data")
    if not isinstance(raw_models, list):
        raise RuntimeManifestError(
            "endpoint.models_readback.data must be a list"
        )
    served_models: set[str] = {
        cast(str, entry.get("id"))
        for entry in raw_models
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and cast(str, entry.get("id"))
    }
    served_model: str = cast(str, endpoint["served_model"])
    if served_model not in served_models:
        raise RuntimeManifestError(
            "Served model is absent from /v1/models readback: "
            f"served_model={served_model}, readback={sorted(served_models)}"
        )
    software: dict[str, JsonValue] = _required_object(
        facts,
        "software",
        "runtime_facts",
    )
    for field_name in (
        "python_version",
        "pytorch_version",
        "transformers_version",
        "cuda_version",
        "driver_version",
    ):
        require_nonempty_string(software.get(field_name), f"software.{field_name}")
    hardware: dict[str, JsonValue] = _required_object(
        facts,
        "hardware",
        "runtime_facts",
    )
    for field_name in ("gpu_model", "gpu_uuid"):
        require_nonempty_string(hardware.get(field_name), f"hardware.{field_name}")
    source: dict[str, JsonValue] = _required_object(
        facts,
        "source",
        "runtime_facts",
    )
    require_nonempty_string(
        source.get("sr_agents_revision"),
        "source.sr_agents_revision",
    )
    return facts


def validate_frozen_k2_runtime_reference(
    runtime_facts: Mapping[str, JsonLike],
) -> FrozenK2RuntimeReference:
    """Require exact equality with the authoritative K=2 model identity."""

    facts: dict[str, JsonValue] = _validate_runtime_facts(runtime_facts)
    job: dict[str, JsonValue] = _required_object(facts, "job", "runtime_facts")
    result_tag: str = require_nonempty_string(
        job.get("result_tag"),
        "job.result_tag",
    )
    reference: FrozenK2RuntimeReference | None = (
        FROZEN_K2_RUNTIME_REFERENCES.get(result_tag)
    )
    if reference is None:
        raise RuntimeManifestError(
            "Runtime facts use an unknown K=2 result tag: "
            f"result_tag={result_tag!r}, "
            f"supported={sorted(FROZEN_K2_RUNTIME_REFERENCES)}"
        )
    checkpoint: dict[str, JsonValue] = _required_object(
        facts,
        "checkpoint",
        "runtime_facts",
    )
    tokenizer: dict[str, JsonValue] = _required_object(
        facts,
        "tokenizer",
        "runtime_facts",
    )
    endpoint: dict[str, JsonValue] = _required_object(
        facts,
        "endpoint",
        "runtime_facts",
    )
    source: dict[str, JsonValue] = _required_object(
        facts,
        "source",
        "runtime_facts",
    )
    expected_fields: tuple[tuple[str, JsonValue, JsonValue | None], ...] = (
        (
            "job.model",
            reference["served_model"],
            job.get("model"),
        ),
        (
            "checkpoint.repository",
            reference["checkpoint_repository"],
            checkpoint.get("repository"),
        ),
        (
            "checkpoint.revision",
            reference["checkpoint_revision"],
            checkpoint.get("revision"),
        ),
        (
            "checkpoint.files_manifest_sha256",
            reference["checkpoint_files_manifest_sha256"],
            checkpoint.get("files_manifest_sha256"),
        ),
        (
            "tokenizer.chat_template_sha256",
            reference["chat_template_sha256"],
            tokenizer.get("chat_template_sha256"),
        ),
        (
            "endpoint.served_model",
            reference["served_model"],
            endpoint.get("served_model"),
        ),
        (
            "endpoint.vllm_version",
            reference["vllm_version"],
            endpoint.get("vllm_version"),
        ),
        ("endpoint.dtype", "bfloat16", endpoint.get("dtype")),
        ("endpoint.quantization", "none", endpoint.get("quantization")),
        ("endpoint.max_model_len", 8192, endpoint.get("max_model_len")),
        (
            "source.sr_agents_revision",
            FROZEN_SRAGENTS_REVISION,
            source.get("sr_agents_revision"),
        ),
    )
    mismatches: list[str] = [
        f"{field_name}:expected={expected!r},observed={observed!r}"
        for field_name, expected, observed in expected_fields
        if expected != observed
    ]
    tokenizer_artifacts: dict[str, JsonValue] = require_json_object(
        tokenizer.get("artifacts"),
        "tokenizer.artifacts",
    )
    for artifact_name, expected_sha256 in reference[
        "tokenizer_artifacts"
    ].items():
        observed_sha256: JsonValue | None = tokenizer_artifacts.get(
            artifact_name
        )
        if observed_sha256 != expected_sha256:
            mismatches.append(
                f"tokenizer.artifacts.{artifact_name}:"
                f"expected={expected_sha256!r},observed={observed_sha256!r}"
            )
    if mismatches:
        raise RuntimeManifestError(
            "Runtime facts do not match the authoritative K=2 identity: "
            f"result_tag={result_tag}, mismatches={mismatches}"
        )
    return reference


def _validate_generation(value: JsonLike) -> dict[str, JsonValue]:
    generation: dict[str, JsonValue] = require_json_object(value, "generation")
    temperature: JsonValue | None = generation.get("temperature")
    if isinstance(temperature, bool) or not isinstance(
        temperature,
        (int, float),
    ):
        raise RuntimeManifestError(
            "generation.temperature must be numeric: "
            f"value={temperature!r}"
        )
    require_positive_integer(generation.get("max_tokens"), "generation.max_tokens")
    if not isinstance(generation.get("thinking"), bool):
        raise RuntimeManifestError(
            "generation.thinking must be Boolean: "
            f"value={generation.get('thinking')!r}"
        )
    if "extra_body" not in generation:
        raise RuntimeManifestError("generation.extra_body is required")
    _json_value(generation["extra_body"], "generation.extra_body")
    return generation


def file_evidence(name: str, path: Path) -> FileEvidence:
    """Build immutable evidence for one named input file."""

    normalized_name: str = require_nonempty_string(name, "artifact.name")
    resolved_path: Path = path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Artifact is not a regular file: name={normalized_name}, path={path}"
        )
    return {
        "name": normalized_name,
        "path": str(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
        "sha256": sha256_file(resolved_path),
    }


def code_file_evidence(
    path: Path,
    repository_root: Path,
) -> CodeFileEvidence:
    """Build immutable evidence for one repository-relative code file."""

    resolved_root: Path = repository_root.resolve()
    resolved_path: Path = path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Code file does not exist: path={resolved_path}")
    try:
        relative_path: str = str(resolved_path.relative_to(resolved_root))
    except ValueError as error:
        raise RuntimeManifestError(
            "Code file is outside the repository root: "
            f"path={resolved_path}, repository_root={resolved_root}"
        ) from error
    return {
        "path": relative_path,
        "size_bytes": resolved_path.stat().st_size,
        "sha256": sha256_file(resolved_path),
    }


def code_bundle_sha256(code_files: Sequence[CodeFileEvidence]) -> str:
    """Hash one strictly sorted path-addressed code member list."""

    payload: list[dict[str, JsonLike]] = [
        {
            "path": entry["path"],
            "size_bytes": entry["size_bytes"],
            "sha256": entry["sha256"],
        }
        for entry in code_files
    ]
    return sha256_json(payload)


def build_job_bound_manifest(
    runtime_facts: Mapping[str, JsonLike],
    generation: Mapping[str, JsonLike],
    artifacts: Sequence[tuple[str, Path]],
    code_paths: Sequence[Path],
    repository_root: Path,
) -> JobBoundManifest:
    """Build one complete credential-free job manifest from explicit inputs."""

    validated_facts: dict[str, JsonValue] = _validate_runtime_facts(runtime_facts)
    validated_generation: dict[str, JsonValue] = _validate_generation(generation)
    if not artifacts:
        raise RuntimeManifestError("At least one named artifact is required")
    artifact_entries: list[FileEvidence] = sorted(
        [file_evidence(name, path) for name, path in artifacts],
        key=lambda entry: entry["name"],
    )
    artifact_names: list[str] = [entry["name"] for entry in artifact_entries]
    if len(artifact_names) != len(set(artifact_names)):
        raise RuntimeManifestError(
            f"Artifact names must be unique: names={artifact_names}"
        )
    if not code_paths:
        raise RuntimeManifestError("At least one code file is required")
    code_entries: list[CodeFileEvidence] = sorted(
        [
            code_file_evidence(path, repository_root)
            for path in code_paths
        ],
        key=lambda entry: entry["path"],
    )
    code_names: list[str] = [entry["path"] for entry in code_entries]
    if len(code_names) != len(set(code_names)):
        raise RuntimeManifestError(
            f"Code file paths must be unique: paths={code_names}"
        )
    manifest: JobBoundManifest = {
        "schema_version": JOB_MANIFEST_SCHEMA_VERSION,
        "runtime_facts": validated_facts,
        "generation": validated_generation,
        "artifacts": artifact_entries,
        "code_files": code_entries,
        "code_bundle_sha256": code_bundle_sha256(code_entries),
    }
    return validate_job_bound_manifest(manifest)


def _validate_file_evidence(
    value: JsonLike,
    context: str,
) -> FileEvidence:
    entry: dict[str, JsonValue] = require_json_object(value, context)
    name: str = require_nonempty_string(entry.get("name"), f"{context}.name")
    path: str = require_nonempty_string(entry.get("path"), f"{context}.path")
    size_bytes: JsonValue | None = entry.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise RuntimeManifestError(
            f"{context}.size_bytes must be a non-negative integer"
        )
    return {
        "name": name,
        "path": path,
        "size_bytes": size_bytes,
        "sha256": require_sha256(entry.get("sha256"), f"{context}.sha256"),
    }


def _validate_code_evidence(
    value: JsonLike,
    context: str,
) -> CodeFileEvidence:
    entry: dict[str, JsonValue] = require_json_object(value, context)
    path: str = require_nonempty_string(entry.get("path"), f"{context}.path")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise RuntimeManifestError(
            f"Code member path must be repository-relative: path={path}"
        )
    size_bytes: JsonValue | None = entry.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise RuntimeManifestError(
            f"{context}.size_bytes must be a non-negative integer"
        )
    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": require_sha256(entry.get("sha256"), f"{context}.sha256"),
    }


def validate_job_bound_manifest(value: JsonLike) -> JobBoundManifest:
    """Validate all manifest identities and recompute its code aggregate."""

    manifest: dict[str, JsonValue] = require_json_object(value, "manifest")
    if manifest.get("schema_version") != JOB_MANIFEST_SCHEMA_VERSION:
        raise RuntimeManifestError(
            "Job manifest schema mismatch: "
            f"expected={JOB_MANIFEST_SCHEMA_VERSION}, "
            f"actual={manifest.get('schema_version')!r}"
        )
    secret_locations: list[str] = _secret_locations(manifest, "$")
    if secret_locations:
        raise RuntimeManifestError(
            "Job manifest contains credential-like material: "
            f"locations={secret_locations}"
        )
    facts: dict[str, JsonValue] = _validate_runtime_facts(
        require_json_object(manifest.get("runtime_facts"), "runtime_facts")
    )
    generation: dict[str, JsonValue] = _validate_generation(
        require_json_object(manifest.get("generation"), "generation")
    )
    raw_artifacts: JsonValue | None = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise RuntimeManifestError("manifest.artifacts must be a non-empty list")
    artifacts: list[FileEvidence] = [
        _validate_file_evidence(entry, f"artifacts[{index}]")
        for index, entry in enumerate(raw_artifacts)
    ]
    artifact_names: list[str] = [entry["name"] for entry in artifacts]
    if artifact_names != sorted(artifact_names) or len(artifact_names) != len(
        set(artifact_names)
    ):
        raise RuntimeManifestError(
            "Manifest artifacts must have unique, strictly sorted names: "
            f"names={artifact_names}"
        )
    raw_code_files: JsonValue | None = manifest.get("code_files")
    if not isinstance(raw_code_files, list) or not raw_code_files:
        raise RuntimeManifestError("manifest.code_files must be a non-empty list")
    code_files: list[CodeFileEvidence] = [
        _validate_code_evidence(entry, f"code_files[{index}]")
        for index, entry in enumerate(raw_code_files)
    ]
    code_names: list[str] = [entry["path"] for entry in code_files]
    if code_names != sorted(code_names) or len(code_names) != len(set(code_names)):
        raise RuntimeManifestError(
            "Manifest code files must have unique, strictly sorted paths: "
            f"paths={code_names}"
        )
    recomputed_bundle: str = code_bundle_sha256(code_files)
    declared_bundle: str = require_sha256(
        manifest.get("code_bundle_sha256"),
        "code_bundle_sha256",
    )
    if recomputed_bundle != declared_bundle:
        raise RuntimeManifestError(
            "Code bundle does not match member evidence: "
            f"declared={declared_bundle}, recomputed={recomputed_bundle}"
        )
    return {
        "schema_version": JOB_MANIFEST_SCHEMA_VERSION,
        "runtime_facts": facts,
        "generation": generation,
        "artifacts": artifacts,
        "code_files": code_files,
        "code_bundle_sha256": declared_bundle,
    }


def load_job_bound_manifest(path: Path) -> JobBoundManifest:
    """Load and validate one job-bound manifest."""

    return validate_job_bound_manifest(
        load_json_file(path, "runtime-manifest")
    )


def verify_job_bound_manifest_files(
    manifest: JobBoundManifest,
    repository_root: Path,
) -> None:
    """Verify all manifest members against the current filesystem."""

    resolved_root: Path = repository_root.resolve()
    if not resolved_root.is_dir():
        raise NotADirectoryError(
            "Manifest repository root does not exist: "
            f"path={resolved_root}"
        )
    for artifact in manifest["artifacts"]:
        artifact_path: Path = Path(artifact["path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(
                "Manifest artifact does not exist: "
                f"name={artifact['name']}, path={artifact_path}"
            )
        observed_size: int = artifact_path.stat().st_size
        observed_sha256: str = sha256_file(artifact_path)
        if (
            observed_size != artifact["size_bytes"]
            or observed_sha256 != artifact["sha256"]
        ):
            raise RuntimeManifestError(
                "Manifest artifact identity mismatch: "
                f"name={artifact['name']}, path={artifact_path}, "
                f"expected_size={artifact['size_bytes']}, "
                f"observed_size={observed_size}, "
                f"expected_sha256={artifact['sha256']}, "
                f"observed_sha256={observed_sha256}"
            )
    observed_code_files: list[CodeFileEvidence] = []
    for member in manifest["code_files"]:
        member_path: Path = resolved_root / member["path"]
        observed_member: CodeFileEvidence = code_file_evidence(
            member_path,
            resolved_root,
        )
        if observed_member != member:
            raise RuntimeManifestError(
                "Manifest code member identity mismatch: "
                f"path={member['path']}, expected={member}, "
                f"observed={observed_member}"
            )
        observed_code_files.append(observed_member)
    observed_bundle: str = code_bundle_sha256(observed_code_files)
    if observed_bundle != manifest["code_bundle_sha256"]:
        raise RuntimeManifestError(
            "Manifest code bundle differs from current filesystem: "
            f"expected={manifest['code_bundle_sha256']}, "
            f"observed={observed_bundle}"
        )


def manifest_artifact(
    manifest: JobBoundManifest,
    name: str,
) -> FileEvidence:
    """Return one uniquely named artifact from a validated manifest."""

    matches: list[FileEvidence] = [
        entry for entry in manifest["artifacts"] if entry["name"] == name
    ]
    if len(matches) != 1:
        raise RuntimeManifestError(
            "Manifest artifact lookup must resolve exactly once: "
            f"name={name}, matches={len(matches)}"
        )
    return matches[0]
