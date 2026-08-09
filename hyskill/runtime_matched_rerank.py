"""Native listwise rerank and answer primitives for matched baselines."""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict, cast

from hyskill.runtime_matched_bm25 import (
    FROZEN_SRAGENTS_REVISION,
    FROZEN_TOP_K,
    git_revision,
    require_module_under,
)
from hyskill.runtime_matched_execution import (
    ExecutionContext,
    FailureCategory,
    JobBoundManifest,
    JsonLike,
    JsonValue,
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
    sha256_json,
    validate_frozen_k2_runtime_reference,
    verify_job_bound_manifest_files,
)


RERANK_DECISION_SCHEMA_VERSION: str = (
    "runtime-matched-rerank-decision-v1"
)
RERANK_DECISION_PAYLOAD_SCHEMA_VERSION: str = (
    "runtime-matched-rerank-payload-v1"
)
RERANK_ANSWER_SCHEMA_VERSION: str = "runtime-matched-baseline-answer-v1"
RERANK_ANSWER_PAYLOAD_SCHEMA_VERSION: str = (
    "runtime-matched-rerank-answer-payload-v1"
)
RERANK_EXECUTION_SCHEMA_VERSION: str = (
    "runtime-matched-rerank-execution-v1"
)
RERANK_ANSWER_EXECUTION_SCHEMA_VERSION: str = (
    "runtime-matched-rerank-answer-execution-v1"
)
RERANK_ARM: str = "always_rerank"
RERANK_DECISION_JOB_ARM: str = RERANK_ARM
RERANK_ANSWER_JOB_ARM: str = RERANK_ARM
RERANK_DECISION_STAGE: str = "decision"
RERANK_ANSWER_STAGE: str = "answer"
RERANK_TEMPERATURE: float = 0.0
RERANK_MAX_TOKENS: int = 1024
RERANK_THINKING: bool = False
RERANK_MAX_PARSE_ATTEMPTS: int = 3
MAX_INFRA_ATTEMPTS: int = 3
ANSWER_TEMPERATURE: float = 0.7
ANSWER_MAX_TOKENS: int = 2048
ANSWER_THINKING: bool = False
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0)
UNSUPPORTED_RESULT_TAGS: frozenset[str] = frozenset(
    {
        "deepseek7b",
        "deepseek-7b",
        "yi15-9b",
        "yi-1.5-9b",
    }
)

JsonObject = dict[str, JsonValue]


class RuntimeMatchedRerankError(ValueError):
    """Raised when a native rerank job violates its frozen contract."""


class FormatCandidates(Protocol):
    """Pinned SR-Agents candidate formatter."""

    def __call__(self, candidates: list[JsonObject]) -> str:
        """Render candidate names and descriptions."""


class ParseRanking(Protocol):
    """Pinned SR-Agents rerank parser."""

    def __call__(
        self,
        response: str,
        candidate_count: int,
    ) -> list[int]:
        """Return unique zero-based candidate indices."""


class BuildPrompt(Protocol):
    """Pinned SR-Agents benchmark prompt builder."""

    def __call__(
        self,
        instance: JsonObject,
        skills: list[str] | None,
    ) -> tuple[str, str]:
        """Build system and user messages."""


class Chat(Protocol):
    """Pinned SR-Agents single-turn chat helper."""

    def __call__(
        self,
        client: OpenAIClientLike,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
        extra_body: JsonObject | None,
    ) -> str:
        """Return one raw model response."""


class CreateClient(Protocol):
    """Pinned SR-Agents OpenAI-compatible client factory."""

    def __call__(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> OpenAIClientLike:
        """Create an endpoint client."""


class GetExtraBody(Protocol):
    """Pinned SR-Agents thinking-control helper."""

    def __call__(
        self,
        model: str,
        thinking: bool,
    ) -> JsonObject | None:
        """Return model-specific request additions."""


class EngineResult(Protocol):
    """Native direct-engine result."""

    raw_output: str
    transcript: str | None
    skill_ids_used: list[str]
    meta: JsonObject


class DirectEngine(Protocol):
    """Native SR-Agents direct-answer engine."""

    def run(
        self,
        instance: JsonObject,
        skills: list[JsonObject],
        client: OpenAIClientLike,
        model: str,
    ) -> EngineResult:
        """Run one direct answer request or tool loop."""


class DirectEngineFactory(Protocol):
    """Native direct-engine constructor."""

    def __call__(
        self,
        *,
        temperature: float,
        max_tokens: int,
        thinking: bool,
    ) -> DirectEngine:
        """Build the frozen answer engine."""


class NativeRerankRuntime(TypedDict):
    """Exact external functions loaded from pinned SR-Agents."""

    prompt_template: str
    format_candidates: FormatCandidates
    parse_ranking: ParseRanking
    build_prompt: BuildPrompt
    chat: Chat
    create_client: CreateClient
    get_extra_body: GetExtraBody
    create_engine: DirectEngineFactory
    request_error_types: tuple[type[Exception], ...]
    revision: str
    source_root: str


class RerankInput(TypedDict):
    """One validated, rendered native rerank request."""

    instance: JsonObject
    candidates: list[JsonObject]
    ordered_candidate_ids: list[str]
    rendered_prompt: str
    candidate_hash: str
    decision_payload_hash: str
    execution_request_hash: str


class RerankOutcome(TypedDict):
    """One rerank outcome before immutable provenance is attached."""

    reranked_candidate_ids: list[str]
    raw_responses: list[str]
    parse_attempts: int
    client_call_attempts: int
    parse_sufficient: bool
    omitted_candidate_count: int
    failure_category: FailureCategory
    error: JsonObject | None


class RerankDecision(TypedDict):
    """Persisted native rerank decision."""

    schema_version: str
    instance_id: str
    domain: str
    arm: str
    stage: str
    model: str
    served_model: str
    ordered_candidate_ids: list[str]
    candidate_hash: str
    decision_payload_hash: str
    execution_request_hash: str
    reranked_candidate_ids: list[str]
    selected_skill_id: str | None
    raw_response: str
    raw_responses: list[str]
    parse_attempts: int
    client_call_attempts: int
    parse_sufficient: bool
    omitted_candidate_count: int
    failure_category: FailureCategory
    runtime_manifest_sha256: str
    code_bundle_sha256: str
    source_sha256: str
    error: JsonObject | None


class AnswerInput(TypedDict):
    """One rendered direct answer bound to a successful rerank decision."""

    instance: JsonObject
    skill: JsonObject
    messages: list[JsonObject]
    tools: list[JsonObject]
    answer_payload_hash: str
    execution_request_hash: str


class AnswerOutcome(TypedDict):
    """One direct-engine outcome before provenance is attached."""

    raw_output: str
    transcript: str | None
    skill_ids_used: list[str]
    meta: JsonObject
    engine_attempts: int
    failure_category: FailureCategory
    error: JsonObject | None


def require_string(value: JsonValue | None, context: str) -> str:
    """Return a non-empty string or raise with context."""

    if not isinstance(value, str) or not value:
        raise RuntimeMatchedRerankError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return a JSON object or raise with context."""

    if not isinstance(value, dict):
        raise RuntimeMatchedRerankError(
            f"Expected JSON object: context={context}, "
            f"value_type={type(value).__name__}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return a JSON list or raise with context."""

    if not isinstance(value, list):
        raise RuntimeMatchedRerankError(
            f"Expected JSON list: context={context}, "
            f"value_type={type(value).__name__}"
        )
    return value


def read_json(path: Path) -> JsonValue:
    """Read one UTF-8 JSON file with precise parse errors."""

    if not path.is_file():
        raise FileNotFoundError(f"Required JSON does not exist: path={path}")
    try:
        return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise RuntimeMatchedRerankError(
            "Malformed JSON: "
            f"path={path}, line={error.lineno}, column={error.colno}, "
            f"message={error.msg}"
        ) from error


def read_jsonl(path: Path) -> list[JsonObject]:
    """Read JSONL rows and reject malformed or duplicate instances."""

    if not path.exists():
        return []
    if not path.is_file():
        raise FileNotFoundError(f"JSONL path is not a file: path={path}")
    rows: list[JsonObject] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeMatchedRerankError(
                    "Malformed JSONL: "
                    f"path={path}, line={line_number}, "
                    f"column={error.colno}, message={error.msg}"
                ) from error
            row: JsonObject = require_object(
                value,
                f"{path}:{line_number}",
            )
            instance_id: str = require_string(
                row.get("instance_id"),
                f"{path}:{line_number}.instance_id",
            )
            if instance_id in seen:
                raise RuntimeMatchedRerankError(
                    "Duplicate JSONL instance: "
                    f"path={path}, instance_id={instance_id}"
                )
            seen.add(instance_id)
            rows.append(row)
    return rows


def append_jsonl(
    path: Path,
    payload: Mapping[str, JsonLike],
    lock: threading.Lock,
) -> None:
    """Append and flush one canonical JSONL record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(canonical_json(payload) + "\n")
            output_file.flush()


def write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, JsonLike]],
) -> None:
    """Atomically replace one canonical JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(canonical_json(row) + "\n")
    temporary_path.replace(path)


def indexed_rows(
    rows: Sequence[JsonObject],
    context: str,
) -> dict[str, JsonObject]:
    """Index rows by a unique instance ID."""

    output: dict[str, JsonObject] = {}
    for index, row in enumerate(rows):
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}[{index}].instance_id",
        )
        if instance_id in output:
            raise RuntimeMatchedRerankError(
                f"Duplicate instance: context={context}, instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def assert_exact_coverage(
    expected_ids: Sequence[str],
    observed_ids: Sequence[str],
    context: str,
) -> None:
    """Require exact ID coverage without duplicates or unexpected rows."""

    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeMatchedRerankError(
            f"Duplicate IDs in coverage input: context={context}"
        )
    expected: set[str] = set(expected_ids)
    observed: set[str] = set(observed_ids)
    if expected != observed:
        raise RuntimeMatchedRerankError(
            f"{context} coverage mismatch: "
            f"missing={sorted(expected - observed)[:20]}, "
            f"unexpected={sorted(observed - expected)[:20]}"
        )


def require_supported_model(model_tag: str) -> None:
    """Reject the two models that cannot fit the 50-candidate prompt."""

    normalized: str = model_tag.strip().lower()
    if normalized in UNSUPPORTED_RESULT_TAGS:
        raise RuntimeMatchedRerankError(
            "Native 50-candidate Rerank is unavailable for this model: "
            f"model_tag={model_tag}"
        )


def load_native_rerank_runtime(
    checkout: Path,
    expected_revision: str,
) -> NativeRerankRuntime:
    """Load exact native prompt/parser/chat/direct-engine behavior."""

    checkout_path: Path = checkout.resolve()
    observed_revision: str = git_revision(checkout_path)
    if observed_revision != expected_revision:
        raise RuntimeError(
            "SR-Agents revision mismatch: "
            f"path={checkout_path}, expected={expected_revision}, "
            f"actual={observed_revision}"
        )
    source_root: Path = checkout_path / "src"
    if not source_root.is_dir():
        raise NotADirectoryError(
            f"SR-Agents source directory does not exist: path={source_root}"
        )
    source_root_text: str = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)
    rerank_module: ModuleType = importlib.import_module(
        "sragents.retrieve.llm_rerank"
    )
    prompts_module: ModuleType = importlib.import_module("sragents.prompts")
    llm_module: ModuleType = importlib.import_module("sragents.llm")
    direct_module: ModuleType = importlib.import_module(
        "sragents.infer.engines.direct"
    )
    openai_module: ModuleType = importlib.import_module("openai")
    for module in (
        rerank_module,
        prompts_module,
        llm_module,
        direct_module,
    ):
        require_module_under(module, source_root)
    prompt_template: object = getattr(
        rerank_module,
        "_RERANK_PROMPT",
        None,
    )
    if not isinstance(prompt_template, str):
        raise RuntimeError(
            "Pinned SR-Agents rerank prompt is unavailable: "
            "attribute=sragents.retrieve.llm_rerank._RERANK_PROMPT"
        )
    api_error_type: object = getattr(openai_module, "APIError", None)
    if not isinstance(api_error_type, type) or not issubclass(
        api_error_type,
        Exception,
    ):
        raise RuntimeError(
            "OpenAI client does not expose APIError for bounded retries"
        )
    return {
        "prompt_template": prompt_template,
        "format_candidates": cast(
            FormatCandidates,
            getattr(rerank_module, "_format_candidates"),
        ),
        "parse_ranking": cast(
            ParseRanking,
            getattr(rerank_module, "_parse_ranking"),
        ),
        "build_prompt": cast(
            BuildPrompt,
            getattr(prompts_module, "build_prompt"),
        ),
        "chat": cast(Chat, getattr(llm_module, "chat")),
        "create_client": cast(
            CreateClient,
            getattr(llm_module, "create_llm_client"),
        ),
        "get_extra_body": cast(
            GetExtraBody,
            getattr(llm_module, "get_extra_body"),
        ),
        "create_engine": cast(
            DirectEngineFactory,
            getattr(direct_module, "DirectEngine"),
        ),
        "request_error_types": (cast(type[Exception], api_error_type),),
        "revision": observed_revision,
        "source_root": str(source_root),
    }


def load_instances(path: Path, domain: str) -> list[JsonObject]:
    """Load and validate one domain's complete instance set."""

    values: list[JsonValue] = require_list(read_json(path), "instances")
    rows: list[JsonObject] = [
        require_object(value, f"instances[{index}]")
        for index, value in enumerate(values)
    ]
    indexed_rows(rows, "instances")
    for row in rows:
        if row.get("dataset") != domain:
            raise RuntimeMatchedRerankError(
                "Instance domain mismatch: "
                f"instance_id={row.get('instance_id')!r}, "
                f"expected={domain}, actual={row.get('dataset')!r}"
            )
    return rows


def load_corpus(path: Path) -> dict[str, JsonObject]:
    """Load a unique skill corpus."""

    values: list[JsonValue] = require_list(read_json(path), "corpus")
    rows: list[JsonObject] = [
        require_object(value, f"corpus[{index}]")
        for index, value in enumerate(values)
    ]
    output: dict[str, JsonObject] = {}
    for index, row in enumerate(rows):
        skill_id: str = require_string(
            row.get("skill_id"),
            f"corpus[{index}].skill_id",
        )
        if skill_id in output:
            raise RuntimeMatchedRerankError(
                f"Corpus contains duplicate skill_id: skill_id={skill_id}"
            )
        output[skill_id] = row
    return output


def load_retrieval_records(path: Path) -> tuple[JsonObject, list[JsonObject]]:
    """Load a standard retrieval artifact with unique records."""

    payload: JsonObject = require_object(read_json(path), "retrieval")
    values: list[JsonValue] = require_list(
        payload.get("results"),
        "retrieval.results",
    )
    rows: list[JsonObject] = [
        require_object(value, f"retrieval.results[{index}]")
        for index, value in enumerate(values)
    ]
    indexed_rows(rows, "retrieval.results")
    return payload, rows


def manifest_generation(
    manifest: JobBoundManifest,
) -> JsonObject:
    """Return the normalized generation block."""

    return dict(manifest["generation"])


def validate_job_manifest(
    manifest_path: Path,
    model_tag: str,
    served_model: str,
    api_base: str,
    domain: str,
    arm: str,
    stage: str,
    expected_generation: Mapping[str, JsonLike],
    required_artifacts: Sequence[tuple[str, Path]],
    required_code_members: Sequence[str],
    repository_root: Path,
) -> JobBoundManifest:
    """Validate a job manifest against CLI identity and immutable inputs."""

    manifest: JobBoundManifest = load_job_bound_manifest(manifest_path)
    verify_job_bound_manifest_files(manifest, repository_root)
    validate_frozen_k2_runtime_reference(manifest["runtime_facts"])
    facts: JsonObject = manifest["runtime_facts"]
    job: JsonObject = require_object(facts.get("job"), "runtime_facts.job")
    expected_job: tuple[tuple[str, str], ...] = (
        ("result_tag", model_tag),
        ("model", served_model),
        ("domain", domain),
        ("arm", arm),
    )
    for field, expected in expected_job:
        if job.get(field) != expected:
            raise RuntimeMatchedRerankError(
                "Runtime job identity mismatch: "
                f"field={field}, expected={expected!r}, actual={job.get(field)!r}"
            )
    if job.get("stage") != stage:
        raise RuntimeMatchedRerankError(
            "Runtime job stage mismatch: "
            f"expected={stage!r}, actual={job.get('stage')!r}"
        )
    endpoint: JsonObject = require_object(
        facts.get("endpoint"),
        "runtime_facts.endpoint",
    )
    endpoint_api_base: str = require_string(
        endpoint.get("api_base"),
        "runtime_facts.endpoint.api_base",
    )
    if endpoint_api_base.rstrip("/") != api_base.rstrip("/"):
        raise RuntimeMatchedRerankError(
            "CLI API base differs from the job-bound endpoint: "
            f"cli={api_base}, manifest={endpoint_api_base}"
        )
    if endpoint.get("max_model_len") != 8192:
        raise RuntimeMatchedRerankError(
            "Runtime-matched baseline requires max_model_len=8192: "
            f"actual={endpoint.get('max_model_len')!r}"
        )
    if str(endpoint.get("dtype", "")).lower() not in (
        "bfloat16",
        "bf16",
    ):
        raise RuntimeMatchedRerankError(
            "Runtime-matched baseline requires BF16: "
            f"actual={endpoint.get('dtype')!r}"
        )
    if str(endpoint.get("quantization", "")).lower() not in (
        "none",
        "null",
    ):
        raise RuntimeMatchedRerankError(
            "Runtime-matched baseline forbids quantization: "
            f"actual={endpoint.get('quantization')!r}"
        )
    source: JsonObject = require_object(
        facts.get("source"),
        "runtime_facts.source",
    )
    if source.get("sr_agents_revision") != FROZEN_SRAGENTS_REVISION:
        raise RuntimeMatchedRerankError(
            "SR-Agents revision mismatch in runtime manifest: "
            f"expected={FROZEN_SRAGENTS_REVISION}, "
            f"actual={source.get('sr_agents_revision')!r}"
        )
    if canonical_json(manifest["generation"]) != canonical_json(
        expected_generation
    ):
        raise RuntimeMatchedRerankError(
            "Generation identity mismatch: "
            f"expected={canonical_json(expected_generation)}, "
            f"actual={canonical_json(manifest['generation'])}"
        )
    for artifact_name, artifact_path in required_artifacts:
        evidence = manifest_artifact(manifest, artifact_name)
        if Path(evidence["path"]).resolve() != artifact_path.resolve():
            raise RuntimeMatchedRerankError(
                "Runtime manifest artifact path mismatch: "
                f"name={artifact_name}, cli_path={artifact_path.resolve()}, "
                f"manifest_path={evidence['path']}"
            )
        observed_size: int = artifact_path.stat().st_size
        observed_sha: str = sha256_file(artifact_path)
        if (
            evidence["size_bytes"] != observed_size
            or evidence["sha256"] != observed_sha
        ):
            raise RuntimeMatchedRerankError(
                "Runtime manifest artifact identity mismatch: "
                f"name={artifact_name}, path={artifact_path}, "
                f"manifest_size={evidence['size_bytes']}, "
                f"observed_size={observed_size}, "
                f"manifest_sha={evidence['sha256']}, "
                f"observed_sha={observed_sha}"
            )
    observed_code_members: set[str] = {
        member["path"] for member in manifest["code_files"]
    }
    missing_code_members: list[str] = sorted(
        set(required_code_members) - observed_code_members
    )
    if missing_code_members:
        raise RuntimeMatchedRerankError(
            "Runtime manifest omits required rerank code members: "
            f"missing={missing_code_members}"
        )
    return manifest


def rerank_generation(extra_body: JsonObject | None) -> JsonObject:
    """Return the exact native rerank generation identity."""

    return {
        "temperature": RERANK_TEMPERATURE,
        "max_tokens": RERANK_MAX_TOKENS,
        "thinking": RERANK_THINKING,
        "extra_body": extra_body,
        "max_parse_attempts": RERANK_MAX_PARSE_ATTEMPTS,
        "omitted_candidate_append": True,
    }


def answer_generation(extra_body: JsonObject | None) -> JsonObject:
    """Return the exact direct-answer generation identity."""

    return {
        "temperature": ANSWER_TEMPERATURE,
        "max_tokens": ANSWER_MAX_TOKENS,
        "thinking": ANSWER_THINKING,
        "extra_body": extra_body,
    }


def build_rerank_inputs(
    instances: Sequence[JsonObject],
    source_records: Mapping[str, JsonObject],
    corpus: Mapping[str, JsonObject],
    runtime: NativeRerankRuntime,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
) -> list[RerankInput]:
    """Render every native listwise request and bind exact hashes."""

    generation: JsonObject = manifest_generation(manifest)
    output: list[RerankInput] = []
    for instance in instances:
        instance_id: str = require_string(
            instance.get("instance_id"),
            "instance.instance_id",
        )
        source_record: JsonObject | None = source_records.get(instance_id)
        if source_record is None:
            raise RuntimeMatchedRerankError(
                f"BM25 source is missing instance: instance_id={instance_id}"
            )
        retrieved_values: list[JsonValue] = require_list(
            source_record.get("retrieved"),
            f"bm25:{instance_id}.retrieved",
        )
        if len(retrieved_values) != FROZEN_TOP_K:
            raise RuntimeMatchedRerankError(
                "BM25 source does not contain exactly 50 candidates: "
                f"instance_id={instance_id}, expected={FROZEN_TOP_K}, "
                f"actual={len(retrieved_values)}"
            )
        candidate_ids: list[str] = []
        candidates: list[JsonObject] = []
        for rank, raw_candidate in enumerate(retrieved_values, start=1):
            candidate: JsonObject = require_object(
                raw_candidate,
                f"bm25:{instance_id}.retrieved[{rank - 1}]",
            )
            skill_id: str = require_string(
                candidate.get("skill_id"),
                f"bm25:{instance_id}.retrieved[{rank - 1}].skill_id",
            )
            if skill_id in candidate_ids:
                raise RuntimeMatchedRerankError(
                    "BM25 source contains a duplicate candidate: "
                    f"instance_id={instance_id}, rank={rank}, skill_id={skill_id}"
                )
            skill: JsonObject | None = corpus.get(skill_id)
            if skill is None:
                raise RuntimeMatchedRerankError(
                    "BM25 candidate is absent from corpus: "
                    f"instance_id={instance_id}, rank={rank}, skill_id={skill_id}"
                )
            candidate_ids.append(skill_id)
            candidates.append(skill)
        _system, query = runtime["build_prompt"](instance, None)
        candidate_text: str = runtime["format_candidates"](candidates)
        prompt: str = runtime["prompt_template"].format(
            query=query,
            candidates=candidate_text,
        )
        candidate_hash: str = sha256_json(
            {
                "ordered_candidate_ids": candidate_ids,
                "formatted_candidates": candidate_text,
            }
        )
        payload_hash: str = answer_payload_hash(
            RERANK_DECISION_PAYLOAD_SCHEMA_VERSION,
            instance,
            [{"role": "user", "content": prompt}],
            [],
            [],
            generation,
        )
        request_hash: str = execution_request_hash(
            RERANK_EXECUTION_SCHEMA_VERSION,
            payload_hash,
            runtime_manifest_sha256,
            manifest["code_bundle_sha256"],
        )
        output.append(
            {
                "instance": instance,
                "candidates": candidates,
                "ordered_candidate_ids": candidate_ids,
                "rendered_prompt": prompt,
                "candidate_hash": candidate_hash,
                "decision_payload_hash": payload_hash,
                "execution_request_hash": request_hash,
            }
        )
    return output


def append_omitted_candidates(
    parsed_indices: Sequence[int],
    ordered_candidate_ids: Sequence[str],
) -> list[str]:
    """Match upstream: parsed order first, then omissions in BM25 order."""

    candidate_count: int = len(ordered_candidate_ids)
    seen: set[int] = set()
    validated: list[int] = []
    for index in parsed_indices:
        if index < 0 or index >= candidate_count:
            raise RuntimeMatchedRerankError(
                "Rerank parser returned an out-of-range index: "
                f"index={index}, candidate_count={candidate_count}"
            )
        if index in seen:
            raise RuntimeMatchedRerankError(
                f"Rerank parser returned a duplicate index: index={index}"
            )
        seen.add(index)
        validated.append(index)
    return [
        ordered_candidate_ids[index]
        for index in (
            validated
            + [
                index
                for index in range(candidate_count)
                if index not in seen
            ]
        )
    ]


def request_rerank_response(
    rerank_input: RerankInput,
    runtime: NativeRerankRuntime,
    client: OpenAIClientLike,
    model: str,
    model_tag: str,
    domain: str,
    job_id: str,
    extra_body: JsonObject | None,
    logical_attempt_start: int,
    sleep: Callable[[float], None],
    retry_delays: Sequence[float],
) -> tuple[str | None, int, FailureCategory, JsonObject | None]:
    """Call one parse attempt with bounded transient retries."""

    instance_id: str = require_string(
        rerank_input["instance"].get("instance_id"),
        "instance.instance_id",
    )
    client_calls: int = 0
    for infra_attempt in range(1, MAX_INFRA_ATTEMPTS + 1):
        client_calls += 1
        logical_attempt: int = logical_attempt_start + client_calls
        context = ExecutionContext(
            job_id,
            model,
            domain,
            RERANK_ARM,
            instance_id,
            logical_attempt,
            rerank_input["decision_payload_hash"],
            rerank_input["execution_request_hash"],
        )
        try:
            with bind_execution_context(context):
                response: str = runtime["chat"](
                    client,
                    model,
                    rerank_input["rendered_prompt"],
                    None,
                    RERANK_TEMPERATURE,
                    RERANK_MAX_TOKENS,
                    None,
                    extra_body,
                )
            return response, client_calls, "success", None
        except runtime["request_error_types"] as error:
            details = error_context(error)
            category: FailureCategory = classify_request_error(
                details.exception_name,
                details.message,
                details.status_code,
                details.response_body,
            )
            should_retry: bool = (
                category == "infra_transient"
                and infra_attempt < MAX_INFRA_ATTEMPTS
            )
            if should_retry:
                warning: JsonObject = {
                    "level": "warning",
                    "event": "rerank_infra_retry",
                    "result_tag": model_tag,
                    "served_model": model,
                    "domain": domain,
                    "instance_id": instance_id,
                    "infra_attempt": infra_attempt,
                    "exception_name": details.exception_name,
                    "status_code": details.status_code,
                    "response_body": details.response_body,
                }
                print(canonical_json(warning), file=sys.stderr, flush=True)
                sleep(retry_delays[infra_attempt - 1])
                continue
            return (
                None,
                client_calls,
                category,
                {
                    "exception_name": details.exception_name,
                    "message": details.message,
                    "status_code": details.status_code,
                    "response_body": details.response_body,
                },
            )
    raise AssertionError("Rerank infrastructure retry loop did not terminate")


def rerank_one(
    rerank_input: RerankInput,
    runtime: NativeRerankRuntime,
    client: OpenAIClientLike,
    model: str,
    model_tag: str,
    domain: str,
    job_id: str,
    extra_body: JsonObject | None,
    sleep: Callable[[float], None],
    retry_delays: Sequence[float],
) -> RerankOutcome:
    """Execute native parse retries and source-order omission append."""

    raw_responses: list[str] = []
    best_indices: list[int] = []
    client_calls: int = 0
    for _parse_attempt in range(1, RERANK_MAX_PARSE_ATTEMPTS + 1):
        response, calls, category, error = request_rerank_response(
            rerank_input,
            runtime,
            client,
            model,
            model_tag,
            domain,
            job_id,
            extra_body,
            client_calls,
            sleep,
            retry_delays,
        )
        client_calls += calls
        if category != "success":
            return {
                "reranked_candidate_ids": [],
                "raw_responses": raw_responses,
                "parse_attempts": len(raw_responses),
                "client_call_attempts": client_calls,
                "parse_sufficient": False,
                "omitted_candidate_count": 0,
                "failure_category": category,
                "error": error,
            }
        if response is None:
            raise AssertionError("Successful rerank call did not return text")
        raw_responses.append(response)
        parsed_indices: list[int] = runtime["parse_ranking"](
            response,
            len(rerank_input["ordered_candidate_ids"]),
        )
        if len(parsed_indices) > len(best_indices):
            best_indices = parsed_indices
        if len(parsed_indices) >= len(
            rerank_input["ordered_candidate_ids"]
        ) // 2:
            break
    reranked: list[str] = append_omitted_candidates(
        best_indices,
        rerank_input["ordered_candidate_ids"],
    )
    return {
        "reranked_candidate_ids": reranked,
        "raw_responses": raw_responses,
        "parse_attempts": len(raw_responses),
        "client_call_attempts": client_calls,
        "parse_sufficient": len(best_indices)
        >= len(rerank_input["ordered_candidate_ids"]) // 2,
        "omitted_candidate_count": (
            len(rerank_input["ordered_candidate_ids"]) - len(best_indices)
        ),
        "failure_category": "success",
        "error": None,
    }


def build_rerank_decision(
    rerank_input: RerankInput,
    outcome: RerankOutcome,
    model_tag: str,
    served_model: str,
    domain: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    source_sha256: str,
) -> RerankDecision:
    """Attach immutable request and runtime identity to one outcome."""

    instance_id: str = require_string(
        rerank_input["instance"].get("instance_id"),
        "instance.instance_id",
    )
    reranked_ids: list[str] = outcome["reranked_candidate_ids"]
    raw_responses: list[str] = outcome["raw_responses"]
    return {
        "schema_version": RERANK_DECISION_SCHEMA_VERSION,
        "instance_id": instance_id,
        "domain": domain,
        "arm": RERANK_ARM,
        "stage": RERANK_DECISION_STAGE,
        "model": model_tag,
        "served_model": served_model,
        "ordered_candidate_ids": list(
            rerank_input["ordered_candidate_ids"]
        ),
        "candidate_hash": rerank_input["candidate_hash"],
        "decision_payload_hash": rerank_input["decision_payload_hash"],
        "execution_request_hash": rerank_input["execution_request_hash"],
        "reranked_candidate_ids": list(reranked_ids),
        "selected_skill_id": reranked_ids[0] if reranked_ids else None,
        "raw_response": raw_responses[-1] if raw_responses else "",
        "raw_responses": list(raw_responses),
        "parse_attempts": outcome["parse_attempts"],
        "client_call_attempts": outcome["client_call_attempts"],
        "parse_sufficient": outcome["parse_sufficient"],
        "omitted_candidate_count": outcome["omitted_candidate_count"],
        "failure_category": outcome["failure_category"],
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "code_bundle_sha256": code_bundle_sha256,
        "source_sha256": source_sha256,
        "error": outcome["error"],
    }


def validate_existing_decision(
    row: JsonObject,
    rerank_input: RerankInput,
    model_tag: str,
    served_model: str,
    domain: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    source_sha256: str,
) -> None:
    """Reject stale or cross-job rows during safe resume."""

    instance_id: str = require_string(
        rerank_input["instance"].get("instance_id"),
        "instance.instance_id",
    )
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("schema_version", RERANK_DECISION_SCHEMA_VERSION),
        ("instance_id", instance_id),
        ("domain", domain),
        ("arm", RERANK_ARM),
        ("stage", RERANK_DECISION_STAGE),
        ("model", model_tag),
        ("served_model", served_model),
        ("ordered_candidate_ids", rerank_input["ordered_candidate_ids"]),
        ("candidate_hash", rerank_input["candidate_hash"]),
        ("decision_payload_hash", rerank_input["decision_payload_hash"]),
        (
            "execution_request_hash",
            rerank_input["execution_request_hash"],
        ),
        ("runtime_manifest_sha256", runtime_manifest_sha256),
        ("code_bundle_sha256", code_bundle_sha256),
        ("source_sha256", source_sha256),
    )
    for field, expected in expected_fields:
        if row.get(field) != expected:
            raise RuntimeMatchedRerankError(
                "Existing rerank decision is stale: "
                f"instance_id={instance_id}, field={field}, "
                f"expected={expected!r}, actual={row.get(field)!r}"
            )
    raw_category: JsonValue | None = row.get("failure_category")
    if raw_category not in (
        "success",
        "infra_transient",
        "method_failure",
        "unclassified_error",
    ):
        raise RuntimeMatchedRerankError(
            "Existing rerank decision has invalid failure category: "
            f"instance_id={instance_id}, category={raw_category!r}"
        )
    raw_reranked: JsonValue | None = row.get("reranked_candidate_ids")
    reranked_values: list[JsonValue] = require_list(
        raw_reranked,
        f"decision:{instance_id}.reranked_candidate_ids",
    )
    reranked_ids: list[str] = [
        require_string(
            value,
            f"decision:{instance_id}.reranked_candidate_ids[{index}]",
        )
        for index, value in enumerate(reranked_values)
    ]
    if raw_category == "success":
        if (
            len(reranked_ids) != FROZEN_TOP_K
            or len(set(reranked_ids)) != FROZEN_TOP_K
            or set(reranked_ids)
            != set(rerank_input["ordered_candidate_ids"])
        ):
            raise RuntimeMatchedRerankError(
                "Successful rerank decision does not preserve the exact "
                "candidate set: "
                f"instance_id={instance_id}, count={len(reranked_ids)}, "
                f"unique={len(set(reranked_ids))}"
            )
        if row.get("selected_skill_id") != reranked_ids[0]:
            raise RuntimeMatchedRerankError(
                "Successful rerank decision selected skill is not top-1: "
                f"instance_id={instance_id}, "
                f"selected={row.get('selected_skill_id')!r}, "
                f"top1={reranked_ids[0]}"
            )
        raw_responses: list[JsonValue] = require_list(
            row.get("raw_responses"),
            f"decision:{instance_id}.raw_responses",
        )
        parse_attempts: JsonValue | None = row.get("parse_attempts")
        if (
            isinstance(parse_attempts, bool)
            or not isinstance(parse_attempts, int)
            or parse_attempts < 1
            or parse_attempts > RERANK_MAX_PARSE_ATTEMPTS
            or len(raw_responses) != parse_attempts
        ):
            raise RuntimeMatchedRerankError(
                "Successful rerank decision has inconsistent parse attempts: "
                f"instance_id={instance_id}, parse_attempts={parse_attempts!r}, "
                f"raw_responses={len(raw_responses)}"
            )
    elif reranked_ids or row.get("selected_skill_id") is not None:
        raise RuntimeMatchedRerankError(
            "Failed rerank decision contains a candidate selection: "
            f"instance_id={instance_id}, "
            f"category={raw_category}, reranked_count={len(reranked_ids)}, "
            f"selected={row.get('selected_skill_id')!r}"
        )


def build_answer_input(
    instance: JsonObject,
    skill: JsonObject,
    runtime: NativeRerankRuntime,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
) -> AnswerInput:
    """Render the exact native direct-engine payload for hashing."""

    content: str = require_string(skill.get("content"), "skill.content")
    system, user = runtime["build_prompt"](instance, [content])
    messages: list[JsonObject] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    raw_tools: JsonValue | None = skill.get("tools")
    tools: list[JsonObject] = []
    if raw_tools is not None:
        tools = [
            require_object(value, f"skill.tools[{index}]")
            for index, value in enumerate(
                require_list(raw_tools, "skill.tools")
            )
        ]
    payload_hash: str = answer_payload_hash(
        RERANK_ANSWER_PAYLOAD_SCHEMA_VERSION,
        instance,
        messages,
        [skill],
        tools,
        manifest_generation(manifest),
    )
    request_hash: str = execution_request_hash(
        RERANK_ANSWER_EXECUTION_SCHEMA_VERSION,
        payload_hash,
        runtime_manifest_sha256,
        manifest["code_bundle_sha256"],
    )
    return {
        "instance": instance,
        "skill": skill,
        "messages": messages,
        "tools": tools,
        "answer_payload_hash": payload_hash,
        "execution_request_hash": request_hash,
    }


def decision_failure_hashes(
    instance: JsonObject,
    decision: JsonObject,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
) -> tuple[str, str]:
    """Hash an explicit zero-call outcome caused by decision failure."""

    payload_hash: str = sha256_json(
        {
            "schema_version": "runtime-matched-rerank-zero-call-v1",
            "instance": instance,
            "decision_failure_category": decision.get("failure_category"),
            "decision_execution_request_hash": decision.get(
                "execution_request_hash"
            ),
            "zero_call_reason": "rerank_decision_method_failure",
        }
    )
    return (
        payload_hash,
        execution_request_hash(
            RERANK_ANSWER_EXECUTION_SCHEMA_VERSION,
            payload_hash,
            runtime_manifest_sha256,
            manifest["code_bundle_sha256"],
        ),
    )


def run_answer_one(
    answer_input: AnswerInput,
    runtime: NativeRerankRuntime,
    engine: DirectEngine,
    client: OpenAIClientLike,
    model: str,
    domain: str,
    job_id: str,
    sleep: Callable[[float], None],
    retry_delays: Sequence[float],
) -> AnswerOutcome:
    """Run one direct answer with exact retry and empty-output behavior."""

    instance_id: str = require_string(
        answer_input["instance"].get("instance_id"),
        "instance.instance_id",
    )
    for engine_attempt in range(1, MAX_INFRA_ATTEMPTS + 1):
        context = ExecutionContext(
            job_id,
            model,
            domain,
            RERANK_ARM,
            instance_id,
            engine_attempt,
            answer_input["answer_payload_hash"],
            answer_input["execution_request_hash"],
        )
        try:
            with bind_execution_context(context):
                result: EngineResult = engine.run(
                    answer_input["instance"],
                    [answer_input["skill"]],
                    client,
                    model,
                )
            if not result.raw_output.strip():
                if engine_attempt < MAX_INFRA_ATTEMPTS:
                    warning: JsonObject = {
                        "level": "warning",
                        "event": "rerank_answer_empty_retry",
                        "model": model,
                        "domain": domain,
                        "instance_id": instance_id,
                        "engine_attempt": engine_attempt,
                    }
                    print(canonical_json(warning), file=sys.stderr, flush=True)
                    sleep(retry_delays[engine_attempt - 1])
                    continue
                return {
                    "raw_output": "",
                    "transcript": result.transcript,
                    "skill_ids_used": [],
                    "meta": dict(result.meta),
                    "engine_attempts": engine_attempt,
                    "failure_category": "method_failure",
                    "error": {
                        "exception_name": "EmptyModelOutput",
                        "message": (
                            "Direct engine returned empty raw_output for all "
                            "bounded attempts"
                        ),
                        "status_code": None,
                        "response_body": "",
                    },
                }
            return {
                "raw_output": result.raw_output,
                "transcript": result.transcript,
                "skill_ids_used": list(result.skill_ids_used),
                "meta": dict(result.meta),
                "engine_attempts": engine_attempt,
                "failure_category": "success",
                "error": None,
            }
        except runtime["request_error_types"] as error:
            details = error_context(error)
            category: FailureCategory = classify_request_error(
                details.exception_name,
                details.message,
                details.status_code,
                details.response_body,
            )
            if (
                category == "infra_transient"
                and engine_attempt < MAX_INFRA_ATTEMPTS
            ):
                warning = {
                    "level": "warning",
                    "event": "rerank_answer_infra_retry",
                    "model": model,
                    "domain": domain,
                    "instance_id": instance_id,
                    "engine_attempt": engine_attempt,
                    "exception_name": details.exception_name,
                    "status_code": details.status_code,
                    "response_body": details.response_body,
                }
                print(canonical_json(warning), file=sys.stderr, flush=True)
                sleep(retry_delays[engine_attempt - 1])
                continue
            return {
                "raw_output": "",
                "transcript": None,
                "skill_ids_used": [],
                "meta": {},
                "engine_attempts": engine_attempt,
                "failure_category": category,
                "error": {
                    "exception_name": details.exception_name,
                    "message": details.message,
                    "status_code": details.status_code,
                    "response_body": details.response_body,
                },
            }
    raise AssertionError("Answer retry loop did not terminate")


def build_answer_record(
    answer_input: AnswerInput,
    decision: JsonObject,
    outcome: AnswerOutcome,
    model_tag: str,
    served_model: str,
    domain: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    decision_source_sha256: str,
) -> JsonObject:
    """Attach decision, request, runtime, and outcome evidence."""

    instance_id: str = require_string(
        answer_input["instance"].get("instance_id"),
        "instance.instance_id",
    )
    selected_skill_id: str = require_string(
        decision.get("selected_skill_id"),
        f"decision:{instance_id}.selected_skill_id",
    )
    record: JsonObject = {
        "schema_version": RERANK_ANSWER_SCHEMA_VERSION,
        "instance_id": instance_id,
        "domain": domain,
        "arm": RERANK_ARM,
        "stage": RERANK_ANSWER_STAGE,
        "model": model_tag,
        "served_model": served_model,
        "raw_output": outcome["raw_output"],
        "skill_ids_used": list(outcome["skill_ids_used"]),
        "expected_skill_ids": [selected_skill_id],
        "answer_payload_hash": answer_input["answer_payload_hash"],
        "execution_request_hash": answer_input["execution_request_hash"],
        "decision_execution_request_hash": decision.get(
            "execution_request_hash"
        ),
        "failure_category": outcome["failure_category"],
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "code_bundle_sha256": code_bundle_sha256,
        "decision_source_sha256": decision_source_sha256,
        "reused_same_arm": False,
        "engine_attempts": outcome["engine_attempts"],
        "zero_call": False,
        "error": outcome["error"],
    }
    if outcome["transcript"] is not None:
        record["transcript"] = outcome["transcript"]
    if outcome["meta"]:
        record["meta"] = outcome["meta"]
    return record


def build_zero_call_answer_record(
    instance: JsonObject,
    decision: JsonObject,
    model_tag: str,
    served_model: str,
    domain: str,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    decision_source_sha256: str,
) -> JsonObject:
    """Emit one method-failure answer row without any model call."""

    instance_id: str = require_string(
        instance.get("instance_id"),
        "instance.instance_id",
    )
    if decision.get("failure_category") != "method_failure":
        raise RuntimeMatchedRerankError(
            "Only decision method failures may become zero-call answers: "
            f"instance_id={instance_id}, "
            f"category={decision.get('failure_category')!r}"
        )
    payload_hash, request_hash = decision_failure_hashes(
        instance,
        decision,
        manifest,
        runtime_manifest_sha256,
    )
    return {
        "schema_version": RERANK_ANSWER_SCHEMA_VERSION,
        "instance_id": instance_id,
        "domain": domain,
        "arm": RERANK_ARM,
        "stage": RERANK_ANSWER_STAGE,
        "model": model_tag,
        "served_model": served_model,
        "raw_output": "",
        "skill_ids_used": [],
        "expected_skill_ids": [],
        "answer_payload_hash": payload_hash,
        "execution_request_hash": request_hash,
        "decision_execution_request_hash": decision.get(
            "execution_request_hash"
        ),
        "failure_category": "method_failure",
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "code_bundle_sha256": manifest["code_bundle_sha256"],
        "decision_source_sha256": decision_source_sha256,
        "reused_same_arm": False,
        "engine_attempts": 0,
        "zero_call": True,
        "zero_call_reason": "rerank_decision_method_failure",
        "error": {
            "exception_name": "RerankDecisionMethodFailure",
            "message": (
                "Answer call was not submitted because native reranking "
                "failed before selecting a skill"
            ),
            "decision_error": decision.get("error"),
        },
    }


def now_sleep(seconds: float) -> None:
    """Sleep for an explicit retry delay."""

    time.sleep(seconds)
