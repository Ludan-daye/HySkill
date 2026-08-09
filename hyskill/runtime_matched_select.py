"""Pure protocol helpers for the runtime-matched BM25 Select baseline."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Literal, Protocol, TypeAlias, TypedDict, cast

from hyskill.runtime_matched_execution import (
    execution_request_hash,
    sha256_json,
)


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
FailureCategory: TypeAlias = Literal[
    "success",
    "selector_fallback",
    "infra_transient",
    "method_failure",
    "unclassified_error",
]

SELECTOR_PAYLOAD_SCHEMA_VERSION: str = (
    "runtime-matched-select-payload-v1"
)
SELECTION_RECORD_SCHEMA_VERSION: str = (
    "runtime-matched-select-decision-v1"
)
SELECT_ARM: str = "select_bm25"
SELECT_STAGE: str = "decision"
POOL_SIZE: int = 50
MAX_PARSE_ATTEMPTS: int = 3
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 64
THINKING: bool = False
SELECT_ELIGIBLE_RESULT_TAGS: frozenset[str] = frozenset(
    {
        "glm4-9b",
        "llama31-8b",
        "mistral7b",
        "qwen3.5-4b-reference",
        "qwen35-9b",
    }
)
SELECT_UNAVAILABLE_RESULT_TAGS: frozenset[str] = frozenset(
    {
        "deepseek7b",
        "yi15-9b",
    }
)


class SelectProtocolError(ValueError):
    """Raised when a Select input violates the frozen baseline protocol."""


class BuildPrompt(Protocol):
    """SR-Agents dataset prompt builder contract."""

    def __call__(
        self,
        instance: JsonObject,
        skills: list[str] | None,
    ) -> tuple[str, str]:
        """Return the native system and user prompts."""


class FormatCandidates(Protocol):
    """SR-Agents native selector candidate formatter contract."""

    def __call__(self, candidates: list[JsonObject]) -> str:
        """Return the ordered name-and-description candidate block."""


class ParseFirstNumber(Protocol):
    """SR-Agents native selector parser contract."""

    def __call__(self, response: str, candidate_count: int) -> int | None:
        """Return a zero-based candidate index or no parsed index."""


class DisplayName(Protocol):
    """SR-Agents non-leaking skill display-name contract."""

    def __call__(
        self,
        skill: JsonObject,
        index: int | None,
    ) -> str:
        """Return one display name without exposing the skill ID."""


class Chat(Protocol):
    """SR-Agents single-turn chat helper contract."""

    def __call__(
        self,
        client: object,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
        extra_body: JsonObject | None,
    ) -> str:
        """Execute one selector request."""


class CreateClient(Protocol):
    """SR-Agents OpenAI-compatible client factory contract."""

    def __call__(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> object:
        """Create one endpoint client."""


class GetExtraBody(Protocol):
    """SR-Agents thinking-control request helper contract."""

    def __call__(
        self,
        model: str,
        thinking: bool,
    ) -> JsonObject | None:
        """Return model-specific request additions."""


class SelectorCall(Protocol):
    """One logical native selector call supplied by the CLI runner."""

    def __call__(self, parse_attempt: int) -> str:
        """Return one successful response or raise SelectorRequestFailure."""


class NativeSelectorRuntime(TypedDict):
    """Exact native selector functions loaded from SR-Agents."""

    prompt_template: str
    build_prompt: BuildPrompt
    format_candidates: FormatCandidates
    parse_first_number: ParseFirstNumber
    display_name: DisplayName
    chat: Chat
    create_client: CreateClient
    get_extra_body: GetExtraBody
    request_error_types: tuple[type[Exception], ...]


class CandidateDisplay(TypedDict):
    """One ordered candidate exactly as identified to the selector."""

    skill_id: str
    name: str
    description: str


class SelectorGeneration(TypedDict):
    """Frozen native selector generation parameters."""

    temperature: float
    max_tokens: int
    thinking: bool
    extra_body: JsonValue
    max_parse_attempts: int
    rank1_fallback: bool


class PreparedSelection(TypedDict):
    """Validated and fully rendered input for one selector decision."""

    instance: JsonObject
    candidates: list[JsonObject]
    candidate_displays: list[CandidateDisplay]
    ordered_candidate_ids: list[str]
    rendered_prompt: str
    candidate_hash: str
    selector_payload_hash: str
    execution_request_hash: str
    generation: SelectorGeneration


class SelectorErrorPayload(TypedDict):
    """Structured request failure context."""

    exception_name: str
    message: str
    status_code: int | None
    response_body: str


class SelectionOutcome(TypedDict):
    """Result of the frozen parse and fallback protocol."""

    selected_rank_zero_based: int | None
    raw_responses: list[str]
    parse_attempts: int
    parse_success: bool
    rank1_fallback: bool
    failure_category: FailureCategory
    error: SelectorErrorPayload | None


class SelectionRecord(TypedDict):
    """Persisted fresh runtime-matched Select decision."""

    schema_version: str
    instance_id: str
    model: str
    served_model: str
    domain: str
    arm: str
    stage: str
    ordered_candidate_ids: list[str]
    candidate_hash: str
    selector_payload_hash: str
    execution_request_hash: str
    selected_skill_id: str | None
    selected_rank: int | None
    raw_response: str
    raw_responses: list[str]
    parse_attempts: int
    client_call_attempts: int
    parse_success: bool
    rank1_fallback: bool
    failure_category: FailureCategory
    runtime_manifest_sha256: str
    code_bundle_sha256: str
    candidate_source_sha256: str
    reused_same_arm: bool
    error: SelectorErrorPayload | None


class SelectorRequestFailure(RuntimeError):
    """A request failure that must never trigger selector rank-1 fallback."""

    def __init__(
        self,
        category: FailureCategory,
        exception_name: str,
        message: str,
        status_code: int | None,
        response_body: str,
    ) -> None:
        super().__init__(message)
        if category not in (
            "infra_transient",
            "method_failure",
            "unclassified_error",
        ):
            raise ValueError(
                "Selector request failure has invalid category: "
                f"category={category}"
            )
        self.category: FailureCategory = category
        self.exception_name: str = exception_name
        self.status_code: int | None = status_code
        self.response_body: str = response_body


def canonical_json(value: object) -> str:
    """Serialize one JSON-compatible value using the frozen canonical form."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SelectProtocolError(
            "Value is not canonical JSON compatible: "
            f"value_type={type(value).__name__}, error={error}"
        ) from error


def require_string(value: JsonValue | None, context: str) -> str:
    """Return one non-empty string with actionable source context."""

    if not isinstance(value, str) or not value:
        raise SelectProtocolError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return one JSON object with actionable source context."""

    if not isinstance(value, dict):
        raise SelectProtocolError(
            "Expected JSON object: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return one JSON list with actionable source context."""

    if not isinstance(value, list):
        raise SelectProtocolError(
            "Expected JSON list: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_select_eligible(result_tag: str) -> None:
    """Reject unsupported and unknown model tags before any endpoint call."""

    if result_tag in SELECT_UNAVAILABLE_RESULT_TAGS:
        raise SelectProtocolError(
            "The frozen 50-candidate Select arm is unavailable for this model: "
            f"result_tag={result_tag}"
        )
    if result_tag not in SELECT_ELIGIBLE_RESULT_TAGS:
        raise SelectProtocolError(
            "Unknown result tag for runtime-matched Select: "
            f"result_tag={result_tag}, "
            f"allowed={sorted(SELECT_ELIGIBLE_RESULT_TAGS)}"
        )


def load_native_selector_runtime() -> NativeSelectorRuntime:
    """Load the exact SR-Agents selector implementation used by K=2."""

    try:
        selector_module: ModuleType = importlib.import_module(
            "sragents.infer.providers.llm_select"
        )
        prompts_module: ModuleType = importlib.import_module(
            "sragents.prompts"
        )
        corpus_module: ModuleType = importlib.import_module(
            "sragents.corpus"
        )
        llm_module: ModuleType = importlib.import_module("sragents.llm")
        openai_module: ModuleType = importlib.import_module("openai")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The frozen SR-Agents selector runtime is unavailable. Install "
            "external/SR-Agents at revision 277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f "
            "inside the project environment before running Select."
        ) from error
    prompt_template: object = getattr(selector_module, "_PROMPT", None)
    if not isinstance(prompt_template, str):
        raise RuntimeError(
            "SR-Agents does not expose the frozen selector prompt: "
            "attribute=sragents.infer.providers.llm_select._PROMPT"
        )
    api_error_type: object = getattr(openai_module, "APIError", None)
    if not isinstance(api_error_type, type) or not issubclass(
        api_error_type,
        Exception,
    ):
        raise RuntimeError(
            "The installed OpenAI client does not expose APIError"
        )
    return {
        "prompt_template": prompt_template,
        "build_prompt": cast(
            BuildPrompt,
            getattr(prompts_module, "build_prompt"),
        ),
        "format_candidates": cast(
            FormatCandidates,
            getattr(selector_module, "_format_candidates"),
        ),
        "parse_first_number": cast(
            ParseFirstNumber,
            getattr(selector_module, "_parse_first_number"),
        ),
        "display_name": cast(
            DisplayName,
            getattr(corpus_module, "display_name"),
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
        "request_error_types": (cast(type[Exception], api_error_type),),
    }


def selector_generation(
    extra_body: JsonObject | None,
) -> SelectorGeneration:
    """Return the immutable K=2-native selector generation identity."""

    normalized_extra_body: JsonValue = cast(
        JsonValue,
        json.loads(canonical_json(extra_body)),
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
    """Build the ordered candidate identity shown to the native selector."""

    output: list[CandidateDisplay] = []
    for rank, candidate in enumerate(candidates, start=1):
        skill_id: str = require_string(
            candidate.get("skill_id"),
            f"candidate[{rank}].skill_id",
        )
        raw_description: JsonValue | None = candidate.get("description", "")
        description: str = (
            raw_description
            if isinstance(raw_description, str)
            else str(raw_description)
        )
        output.append(
            {
                "skill_id": skill_id,
                "name": display_name(candidate, rank),
                "description": description,
            }
        )
    return output


def source_candidate_ids(source_record: JsonObject) -> list[str]:
    """Return exactly 50 unique ordered IDs from one frozen BM25 record."""

    instance_id: str = require_string(
        source_record.get("instance_id"),
        "source.instance_id",
    )
    raw_retrieved: list[JsonValue] = require_list(
        source_record.get("retrieved"),
        f"source:{instance_id}.retrieved",
    )
    if len(raw_retrieved) != POOL_SIZE:
        raise SelectProtocolError(
            "BM25 source must contain exactly the frozen 50 candidates: "
            f"instance_id={instance_id}, expected={POOL_SIZE}, "
            f"actual={len(raw_retrieved)}"
        )
    candidate_ids: list[str] = []
    for rank, raw_candidate in enumerate(raw_retrieved, start=1):
        candidate: JsonObject = require_object(
            raw_candidate,
            f"source:{instance_id}.retrieved[{rank}]",
        )
        candidate_ids.append(
            require_string(
                candidate.get("skill_id"),
                f"source:{instance_id}.retrieved[{rank}].skill_id",
            )
        )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SelectProtocolError(
            "BM25 source contains duplicate candidate IDs: "
            f"instance_id={instance_id}"
        )
    return candidate_ids


def prepare_selection(
    instance: JsonObject,
    source_record: JsonObject,
    corpus: Mapping[str, JsonObject],
    runtime: NativeSelectorRuntime,
    served_model: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
) -> PreparedSelection:
    """Render one BM25 selector request exactly like K=2 Hy+Select."""

    instance_id: str = require_string(
        instance.get("instance_id"),
        "instance.instance_id",
    )
    source_instance_id: str = require_string(
        source_record.get("instance_id"),
        "source.instance_id",
    )
    if source_instance_id != instance_id:
        raise SelectProtocolError(
            "Instance and BM25 source IDs differ: "
            f"instance_id={instance_id}, source_instance_id={source_instance_id}"
        )
    ordered_candidate_ids: list[str] = source_candidate_ids(source_record)
    missing_ids: list[str] = [
        skill_id
        for skill_id in ordered_candidate_ids
        if skill_id not in corpus
    ]
    if missing_ids:
        raise SelectProtocolError(
            "BM25 candidates are absent from the frozen corpus: "
            f"instance_id={instance_id}, sample={missing_ids[:20]}"
        )
    candidates: list[JsonObject] = [
        corpus[skill_id] for skill_id in ordered_candidate_ids
    ]
    displays: list[CandidateDisplay] = candidate_displays(
        candidates,
        runtime["display_name"],
    )
    _, query = runtime["build_prompt"](instance, None)
    rendered_prompt: str = runtime["prompt_template"].format(
        query=query,
        candidates=runtime["format_candidates"](candidates),
    )
    generation: SelectorGeneration = selector_generation(
        runtime["get_extra_body"](served_model, THINKING)
    )
    payload: JsonObject = {
        "schema_version": SELECTOR_PAYLOAD_SCHEMA_VERSION,
        "arm": SELECT_ARM,
        "instance_id": instance_id,
        "instance": instance,
        "rendered_messages": [
            {
                "role": "user",
                "content": rendered_prompt,
            }
        ],
        "ordered_candidate_ids": list(ordered_candidate_ids),
        "candidate_displays": cast(list[JsonValue], displays),
        "generation": cast(JsonObject, generation),
    }
    selector_payload_sha256: str = sha256_json(payload)
    return {
        "instance": instance,
        "candidates": candidates,
        "candidate_displays": displays,
        "ordered_candidate_ids": ordered_candidate_ids,
        "rendered_prompt": rendered_prompt,
        "candidate_hash": sha256_json(displays),
        "selector_payload_hash": selector_payload_sha256,
        "execution_request_hash": execution_request_hash(
            SELECTOR_PAYLOAD_SCHEMA_VERSION,
            selector_payload_sha256,
            runtime_manifest_sha256,
            code_bundle_sha256,
        ),
        "generation": generation,
    }


def run_selection_protocol(
    selector_call: SelectorCall,
    parse_first_number: ParseFirstNumber,
    candidate_count: int,
) -> SelectionOutcome:
    """Apply three native parse attempts and parse-only rank-1 fallback."""

    if candidate_count != POOL_SIZE:
        raise SelectProtocolError(
            "Selector protocol requires exactly 50 candidates: "
            f"expected={POOL_SIZE}, actual={candidate_count}"
        )
    raw_responses: list[str] = []
    for parse_attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
        try:
            response: str = selector_call(parse_attempt)
        except SelectorRequestFailure as error:
            return {
                "selected_rank_zero_based": None,
                "raw_responses": raw_responses,
                "parse_attempts": len(raw_responses),
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
        selected_index: int | None = parse_first_number(
            response,
            candidate_count,
        )
        if selected_index is not None:
            return {
                "selected_rank_zero_based": selected_index,
                "raw_responses": raw_responses,
                "parse_attempts": parse_attempt,
                "parse_success": True,
                "rank1_fallback": False,
                "failure_category": "success",
                "error": None,
            }
    return {
        "selected_rank_zero_based": 0,
        "raw_responses": raw_responses,
        "parse_attempts": MAX_PARSE_ATTEMPTS,
        "parse_success": False,
        "rank1_fallback": True,
        "failure_category": "selector_fallback",
        "error": None,
    }


def build_selection_record(
    prepared: PreparedSelection,
    outcome: SelectionOutcome,
    result_tag: str,
    served_model: str,
    domain: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    candidate_source_sha256: str,
    client_call_attempts: int,
) -> SelectionRecord:
    """Attach immutable fresh-run provenance to one selector outcome."""

    require_select_eligible(result_tag)
    instance_id: str = require_string(
        prepared["instance"].get("instance_id"),
        "instance.instance_id",
    )
    instance_domain: str = require_string(
        prepared["instance"].get("dataset"),
        f"instance:{instance_id}.dataset",
    )
    if instance_domain != domain:
        raise SelectProtocolError(
            "Selector domain does not match instance: "
            f"instance_id={instance_id}, expected={domain}, "
            f"actual={instance_domain}"
        )
    selected_index: int | None = outcome["selected_rank_zero_based"]
    selected_skill_id: str | None = (
        prepared["ordered_candidate_ids"][selected_index]
        if selected_index is not None
        else None
    )
    raw_responses: list[str] = outcome["raw_responses"]
    if client_call_attempts < len(raw_responses):
        raise SelectProtocolError(
            "Client call count is smaller than successful responses: "
            f"instance_id={instance_id}, calls={client_call_attempts}, "
            f"responses={len(raw_responses)}"
        )
    return {
        "schema_version": SELECTION_RECORD_SCHEMA_VERSION,
        "instance_id": instance_id,
        "model": result_tag,
        "served_model": served_model,
        "domain": domain,
        "arm": SELECT_ARM,
        "stage": SELECT_STAGE,
        "ordered_candidate_ids": list(
            prepared["ordered_candidate_ids"]
        ),
        "candidate_hash": prepared["candidate_hash"],
        "selector_payload_hash": prepared["selector_payload_hash"],
        "execution_request_hash": prepared["execution_request_hash"],
        "selected_skill_id": selected_skill_id,
        "selected_rank": (
            selected_index + 1 if selected_index is not None else None
        ),
        "raw_response": raw_responses[-1] if raw_responses else "",
        "raw_responses": list(raw_responses),
        "parse_attempts": outcome["parse_attempts"],
        "client_call_attempts": client_call_attempts,
        "parse_success": outcome["parse_success"],
        "rank1_fallback": outcome["rank1_fallback"],
        "failure_category": outcome["failure_category"],
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "code_bundle_sha256": code_bundle_sha256,
        "candidate_source_sha256": candidate_source_sha256,
        "reused_same_arm": False,
        "error": outcome["error"],
    }


def selected_skill_ids(record: Mapping[str, JsonValue]) -> tuple[str, ...]:
    """Resolve the one selected skill or an explicit decision-stage failure."""

    category: JsonValue | None = record.get("failure_category")
    selected_skill_id: JsonValue | None = record.get("selected_skill_id")
    if category in ("success", "selector_fallback"):
        if not isinstance(selected_skill_id, str) or not selected_skill_id:
            raise SelectProtocolError(
                "Successful Select decision has no selected skill: "
                f"instance_id={record.get('instance_id')!r}, "
                f"failure_category={category!r}"
            )
        return (selected_skill_id,)
    if category == "method_failure":
        if selected_skill_id is not None:
            raise SelectProtocolError(
                "Failed Select decision unexpectedly selected a skill: "
                f"instance_id={record.get('instance_id')!r}, "
                f"selected_skill_id={selected_skill_id!r}"
            )
        return ()
    raise SelectProtocolError(
        "Select decision contains an unresolved failure: "
        f"instance_id={record.get('instance_id')!r}, "
        f"failure_category={category!r}"
    )
