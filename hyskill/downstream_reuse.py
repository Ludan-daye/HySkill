"""Pure helpers for K=2 downstream request identity and reuse auditing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypeAlias, TypedDict, cast


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
SemanticArm: TypeAlias = Literal[
    "routed_always",
    "routed_gated",
    "routed_select",
    "fixed_gated",
]
ReuseStatus: TypeAlias = Literal[
    "reused_same_arm",
    "needs_inference",
    "rejected",
]

FAILURE_CATEGORIES: frozenset[str] = frozenset(
    {
        "success",
        "selector_fallback",
        "infra_transient",
        "method_failure",
        "unclassified_error",
    }
)
SEMANTIC_ARMS: frozenset[str] = frozenset(
    {
        "routed_always",
        "routed_gated",
        "routed_select",
        "fixed_gated",
    }
)
SECRET_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
    }
)
SHA256_PATTERN: re.Pattern[str] = re.compile(r"[0-9a-f]{64}")
RESULT_TAGS: frozenset[str] = frozenset(
    {
        "qwen3.5-4b-reference",
        "qwen35-9b",
        "mistral7b",
        "deepseek7b",
        "glm4-9b",
        "llama31-8b",
        "yi15-9b",
    }
)
SELECT_ELIGIBLE_FLEET_TAGS: frozenset[str] = frozenset(
    {
        "qwen35-9b",
        "mistral7b",
        "glm4-9b",
        "llama31-8b",
    }
)


class CodeFileDigest(TypedDict):
    """One repository-relative code file and its immutable digest."""

    path: str
    sha256: str


class RuntimeManifestRequired(TypedDict):
    """Immutable request-relevant model, runtime, input, and code identity."""

    schema_version: str
    instances_sha256: str
    corpus_sha256: str
    runtime_identity: dict[str, JsonValue]
    answer_code_bundle_sha256: str
    selector_code_bundle_sha256: str


class RuntimeManifest(RuntimeManifestRequired, total=False):
    """Runtime identity with optional v2 code and legacy provenance."""

    answer_code_files: list[CodeFileDigest]
    selector_code_files: list[CodeFileDigest]
    legacy_jsonl_sha256: str
    legacy_jsonl_records: int
    legacy_result_tag: str
    legacy_semantic_arm: SemanticArm
    legacy_method_label: str


class LegacyManifestEvidence(TypedDict):
    """Immutable binding from one old runtime manifest to one legacy JSONL."""

    legacy_jsonl_sha256: str
    legacy_jsonl_records: int
    legacy_result_tag: str
    legacy_semantic_arm: SemanticArm
    legacy_method_label: str


class SelectorGeneration(TypedDict):
    """Frozen selector generation parameters."""

    temperature: float
    max_tokens: int
    thinking: bool
    extra_body: JsonValue
    max_parse_attempts: int
    rank1_fallback: bool


class AnswerGeneration(TypedDict):
    """Frozen direct-answer generation parameters."""

    temperature: float
    max_tokens: int
    thinking: bool
    extra_body: JsonValue


class CandidateDisplay(TypedDict):
    """One ordered selector candidate as seen by the selector."""

    skill_id: str
    name: str
    description: str


@dataclass(frozen=True)
class CoverageAudit:
    """Expected-versus-observed record coverage."""

    expected: int
    observed: int
    missing_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    unexpected_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Return whether coverage is exact and duplicate-free."""

        return not self.missing_ids and not self.duplicate_ids and not self.unexpected_ids


@dataclass(frozen=True)
class PreseedEligibility:
    """Pure result of one same-arm preseed decision."""

    eligible: bool
    reason: str


class DownstreamDataError(ValueError):
    """Raised when downstream data cannot satisfy the frozen protocol."""


def _json_value(value: JsonLike, location: str) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DownstreamDataError(
                f"Non-finite JSON number: location={location}, value={value}"
            )
        return value
    if isinstance(value, Mapping):
        output: dict[str, JsonValue] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise DownstreamDataError(
                    "JSON object key must be a string: "
                    f"location={location}, key_type={type(raw_key).__name__}"
                )
            output[raw_key] = _json_value(raw_value, f"{location}.{raw_key}")
        return output
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _json_value(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise DownstreamDataError(
        "Unsupported canonical JSON value: "
        f"location={location}, value_type={type(value).__name__}"
    )


def canonical_json(value: JsonLike) -> str:
    """Serialize a JSON-compatible value using the frozen canonical format."""

    normalized: JsonValue = _json_value(value, "$")
    return json.dumps(
        normalized,
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


def code_file_digests(
    paths: Sequence[Path],
    repository_root: Path,
) -> list[CodeFileDigest]:
    """Return the strictly sorted member list for one code bundle."""

    resolved_root: Path = repository_root.resolve()
    if not paths:
        raise DownstreamDataError("Code bundle must contain at least one file")
    entries: list[CodeFileDigest] = []
    seen_paths: set[str] = set()
    for path in paths:
        resolved_path: Path = path.resolve()
        try:
            relative_path: str = str(resolved_path.relative_to(resolved_root))
        except ValueError as error:
            raise DownstreamDataError(
                "Code bundle file is outside repository root: "
                f"path={resolved_path}, repository_root={resolved_root}"
            ) from error
        if relative_path in seen_paths:
            raise DownstreamDataError(
                f"Code bundle contains a duplicate path: path={relative_path}"
            )
        seen_paths.add(relative_path)
        entries.append(
            {
                "path": relative_path,
                "sha256": sha256_file(resolved_path),
            }
        )
    return sorted(entries, key=lambda item: item["path"])


def code_bundle_sha256_from_digests(
    digests: Sequence[CodeFileDigest],
) -> str:
    """Hash a validated path-addressed code member list."""

    payload: list[dict[str, JsonValue]] = [
        {
            "path": digest["path"],
            "sha256": digest["sha256"],
        }
        for digest in digests
    ]
    return sha256_json(payload)


def code_bundle_sha256(paths: Sequence[Path], repository_root: Path) -> str:
    """Hash a strictly sorted, path-addressed set of source files."""

    return code_bundle_sha256_from_digests(
        code_file_digests(paths, repository_root)
    )


def _require_sha256(value: JsonValue | None, field_name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise DownstreamDataError(
            "Field must be a lowercase SHA-256 digest: "
            f"field={field_name}, value={value!r}"
        )
    return value


def _validate_code_file_path(value: JsonValue | None, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DownstreamDataError(
            "Code member path must be a non-empty string: "
            f"field={field_name}, value={value!r}"
        )
    normalized: PurePosixPath = PurePosixPath(value)
    if (
        normalized.is_absolute()
        or value == "."
        or ".." in normalized.parts
        or "\\" in value
        or str(normalized) != value
    ):
        raise DownstreamDataError(
            "Code member path must be normalized and repository-relative: "
            f"field={field_name}, value={value!r}"
        )
    return value


def validate_code_file_digests(
    value: JsonValue | None,
    field_name: str,
) -> list[CodeFileDigest]:
    """Validate one non-empty, unique, strictly sorted member list."""

    if not isinstance(value, list) or not value:
        raise DownstreamDataError(
            "Code member list must be a non-empty JSON list: "
            f"field={field_name}, value_type={type(value).__name__}"
        )
    output: list[CodeFileDigest] = []
    for index, raw_entry in enumerate(value):
        if not isinstance(raw_entry, dict):
            raise DownstreamDataError(
                "Code member must be a JSON object: "
                f"field={field_name}, index={index}, "
                f"value_type={type(raw_entry).__name__}"
            )
        if set(raw_entry) != {"path", "sha256"}:
            raise DownstreamDataError(
                "Code member must contain exactly path and sha256: "
                f"field={field_name}, index={index}, keys={sorted(raw_entry)}"
            )
        output.append(
            {
                "path": _validate_code_file_path(
                    raw_entry.get("path"),
                    f"{field_name}[{index}].path",
                ),
                "sha256": _require_sha256(
                    raw_entry.get("sha256"),
                    f"{field_name}[{index}].sha256",
                ),
            }
        )
    paths: list[str] = [entry["path"] for entry in output]
    if len(set(paths)) != len(paths):
        raise DownstreamDataError(
            f"Code member list contains duplicate paths: field={field_name}"
        )
    if paths != sorted(paths):
        raise DownstreamDataError(
            f"Code member list is not strictly sorted: field={field_name}"
        )
    return output


def _secret_locations(value: JsonValue, location: str) -> list[str]:
    if isinstance(value, dict):
        locations: list[str] = []
        for key, child in value.items():
            normalized_key: str = key.lower().replace("-", "_")
            secret_suffixes: tuple[str, ...] = (
                "_api_key",
                "_password",
                "_secret",
                "_access_token",
                "_refresh_token",
            )
            if normalized_key in SECRET_KEYS or normalized_key.endswith(
                secret_suffixes
            ):
                locations.append(f"{location}.{key}")
            locations.extend(_secret_locations(child, f"{location}.{key}"))
        return locations
    if isinstance(value, list):
        output: list[str] = []
        for index, child in enumerate(value):
            output.extend(_secret_locations(child, f"{location}[{index}]"))
        return output
    return []


def validate_runtime_manifest(
    value: JsonLike,
    instances_sha256: str,
    corpus_sha256: str,
) -> RuntimeManifest:
    """Validate the common runtime-manifest schema and input identity."""

    normalized: JsonValue = _json_value(value, "$")
    if not isinstance(normalized, dict):
        raise DownstreamDataError(
            "Runtime manifest must be a JSON object: "
            f"value_type={type(normalized).__name__}"
        )
    secret_locations: list[str] = _secret_locations(normalized, "$")
    if secret_locations:
        raise DownstreamDataError(
            "Runtime manifest contains credential-like keys: "
            f"locations={secret_locations}"
        )
    required_keys: tuple[str, ...] = (
        "schema_version",
        "instances_sha256",
        "corpus_sha256",
        "runtime_identity",
        "answer_code_bundle_sha256",
        "selector_code_bundle_sha256",
    )
    missing_keys: list[str] = [
        key for key in required_keys if key not in normalized
    ]
    if missing_keys:
        raise DownstreamDataError(
            f"Runtime manifest is missing required keys: keys={missing_keys}"
        )
    schema_version: JsonValue = normalized["schema_version"]
    if not isinstance(schema_version, str) or not schema_version:
        raise DownstreamDataError(
            "Runtime manifest field must be a non-empty string: "
            f"field=schema_version, value={schema_version!r}"
        )
    normalized_instances_sha256: str = _require_sha256(
        normalized["instances_sha256"],
        "instances_sha256",
    )
    normalized_corpus_sha256: str = _require_sha256(
        normalized["corpus_sha256"],
        "corpus_sha256",
    )
    answer_bundle_sha256: str = _require_sha256(
        normalized["answer_code_bundle_sha256"],
        "answer_code_bundle_sha256",
    )
    selector_bundle_sha256: str = _require_sha256(
        normalized["selector_code_bundle_sha256"],
        "selector_code_bundle_sha256",
    )
    runtime_identity: JsonValue = normalized["runtime_identity"]
    if not isinstance(runtime_identity, dict) or not runtime_identity:
        raise DownstreamDataError(
            "Runtime manifest runtime_identity must be a non-empty object"
        )
    if normalized_instances_sha256 != instances_sha256:
        raise DownstreamDataError(
            "Runtime manifest instances hash mismatch: "
            f"expected={instances_sha256}, actual={normalized_instances_sha256}"
        )
    if normalized_corpus_sha256 != corpus_sha256:
        raise DownstreamDataError(
            "Runtime manifest corpus hash mismatch: "
            f"expected={corpus_sha256}, actual={normalized_corpus_sha256}"
        )
    answer_members_present: bool = "answer_code_files" in normalized
    selector_members_present: bool = "selector_code_files" in normalized
    if answer_members_present != selector_members_present:
        raise DownstreamDataError(
            "Runtime manifest must contain both code member lists or neither"
        )
    answer_code_files: list[CodeFileDigest] | None = None
    selector_code_files: list[CodeFileDigest] | None = None
    if answer_members_present:
        answer_code_files = validate_code_file_digests(
            normalized.get("answer_code_files"),
            "answer_code_files",
        )
        selector_code_files = validate_code_file_digests(
            normalized.get("selector_code_files"),
            "selector_code_files",
        )
        recomputed_answer_hash: str = code_bundle_sha256_from_digests(
            answer_code_files
        )
        recomputed_selector_hash: str = code_bundle_sha256_from_digests(
            selector_code_files
        )
        if recomputed_answer_hash != answer_bundle_sha256:
            raise DownstreamDataError(
                "Answer code aggregate does not match its member list: "
                f"expected={answer_bundle_sha256}, "
                f"recomputed={recomputed_answer_hash}"
            )
        if recomputed_selector_hash != selector_bundle_sha256:
            raise DownstreamDataError(
                "Selector code aggregate does not match its member list: "
                f"expected={selector_bundle_sha256}, "
                f"recomputed={recomputed_selector_hash}"
            )
    legacy_keys: tuple[str, ...] = (
        "legacy_jsonl_sha256",
        "legacy_jsonl_records",
        "legacy_result_tag",
        "legacy_semantic_arm",
        "legacy_method_label",
    )
    present_legacy_keys: list[str] = [
        key for key in legacy_keys if key in normalized
    ]
    if present_legacy_keys and len(present_legacy_keys) != len(legacy_keys):
        missing_legacy_keys: list[str] = [
            key for key in legacy_keys if key not in normalized
        ]
        raise DownstreamDataError(
            "Legacy manifest evidence is incomplete: "
            f"missing={missing_legacy_keys}"
        )
    legacy_evidence: LegacyManifestEvidence | None = None
    if present_legacy_keys:
        legacy_records_value: JsonValue = normalized["legacy_jsonl_records"]
        if (
            isinstance(legacy_records_value, bool)
            or not isinstance(legacy_records_value, int)
            or legacy_records_value < 0
        ):
            raise DownstreamDataError(
                "legacy_jsonl_records must be a non-negative integer: "
                f"value={legacy_records_value!r}"
            )
        legacy_result_tag_value: JsonValue = normalized["legacy_result_tag"]
        if (
            not isinstance(legacy_result_tag_value, str)
            or not legacy_result_tag_value
        ):
            raise DownstreamDataError(
                "legacy_result_tag must be a non-empty string: "
                f"value={legacy_result_tag_value!r}"
            )
        legacy_arm_value: JsonValue = normalized["legacy_semantic_arm"]
        if not isinstance(legacy_arm_value, str) or legacy_arm_value not in SEMANTIC_ARMS:
            raise DownstreamDataError(
                "legacy_semantic_arm is invalid: "
                f"value={legacy_arm_value!r}, allowed={sorted(SEMANTIC_ARMS)}"
            )
        legacy_method_value: JsonValue = normalized["legacy_method_label"]
        if not isinstance(legacy_method_value, str) or not legacy_method_value:
            raise DownstreamDataError(
                "legacy_method_label must be a non-empty string: "
                f"value={legacy_method_value!r}"
            )
        legacy_evidence = {
            "legacy_jsonl_sha256": _require_sha256(
                normalized["legacy_jsonl_sha256"],
                "legacy_jsonl_sha256",
            ),
            "legacy_jsonl_records": legacy_records_value,
            "legacy_result_tag": legacy_result_tag_value,
            "legacy_semantic_arm": cast(SemanticArm, legacy_arm_value),
            "legacy_method_label": legacy_method_value,
        }
    output: RuntimeManifest = {
        "schema_version": schema_version,
        "instances_sha256": normalized_instances_sha256,
        "corpus_sha256": normalized_corpus_sha256,
        "runtime_identity": runtime_identity,
        "answer_code_bundle_sha256": answer_bundle_sha256,
        "selector_code_bundle_sha256": selector_bundle_sha256,
    }
    if answer_code_files is not None and selector_code_files is not None:
        output["answer_code_files"] = answer_code_files
        output["selector_code_files"] = selector_code_files
    if legacy_evidence is not None:
        output["legacy_jsonl_sha256"] = legacy_evidence[
            "legacy_jsonl_sha256"
        ]
        output["legacy_jsonl_records"] = legacy_evidence[
            "legacy_jsonl_records"
        ]
        output["legacy_result_tag"] = legacy_evidence["legacy_result_tag"]
        output["legacy_semantic_arm"] = legacy_evidence[
            "legacy_semantic_arm"
        ]
        output["legacy_method_label"] = legacy_evidence[
            "legacy_method_label"
        ]
    return output


def require_runtime_code_files(
    manifest: RuntimeManifest,
    context: str,
) -> tuple[list[CodeFileDigest], list[CodeFileDigest]]:
    """Require v2 member evidence before any legacy answer reuse."""

    answer_code_files: list[CodeFileDigest] | None = manifest.get(
        "answer_code_files"
    )
    selector_code_files: list[CodeFileDigest] | None = manifest.get(
        "selector_code_files"
    )
    if answer_code_files is None or selector_code_files is None:
        raise DownstreamDataError(
            "Runtime manifest lacks code member evidence required for reuse: "
            f"context={context}"
        )
    return answer_code_files, selector_code_files


def validate_legacy_manifest_evidence(
    manifest: RuntimeManifest,
    legacy_jsonl_sha256: str,
    legacy_jsonl_records: int,
    result_tag: str,
    semantic_arm: SemanticArm,
    method_label: str,
) -> LegacyManifestEvidence:
    """Validate an old manifest's immutable binding to one legacy source."""

    require_runtime_code_files(manifest, "old-runtime-manifest")
    expected: LegacyManifestEvidence = {
        "legacy_jsonl_sha256": legacy_jsonl_sha256,
        "legacy_jsonl_records": legacy_jsonl_records,
        "legacy_result_tag": result_tag,
        "legacy_semantic_arm": semantic_arm,
        "legacy_method_label": method_label,
    }
    fields: tuple[tuple[str, JsonLike | None, JsonLike], ...] = (
        (
            "legacy_jsonl_sha256",
            manifest.get("legacy_jsonl_sha256"),
            legacy_jsonl_sha256,
        ),
        (
            "legacy_jsonl_records",
            manifest.get("legacy_jsonl_records"),
            legacy_jsonl_records,
        ),
        (
            "legacy_result_tag",
            manifest.get("legacy_result_tag"),
            result_tag,
        ),
        (
            "legacy_semantic_arm",
            manifest.get("legacy_semantic_arm"),
            semantic_arm,
        ),
        (
            "legacy_method_label",
            manifest.get("legacy_method_label"),
            method_label,
        ),
    )
    for field_name, actual_value, expected_value in fields:
        if canonical_json(actual_value) != canonical_json(expected_value):
            raise DownstreamDataError(
                "Legacy manifest evidence mismatch: "
                f"field={field_name}, expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )
    return expected


def validate_selector_runtime_manifest(
    value: JsonLike,
    instances_sha256: str,
    corpus_sha256: str,
    code_bundle_sha256_value: str,
) -> RuntimeManifest:
    """Validate a runtime manifest for selector requests."""

    manifest: RuntimeManifest = validate_runtime_manifest(
        value,
        instances_sha256,
        corpus_sha256,
    )
    if manifest["selector_code_bundle_sha256"] != code_bundle_sha256_value:
        raise DownstreamDataError(
            "Selector code bundle hash mismatch: "
            f"expected={code_bundle_sha256_value}, "
            f"actual={manifest['selector_code_bundle_sha256']}"
        )
    return manifest


def validate_answer_runtime_manifest(
    value: JsonLike,
    instances_sha256: str,
    corpus_sha256: str,
    code_bundle_sha256_value: str,
) -> RuntimeManifest:
    """Validate a runtime manifest for direct-answer requests."""

    manifest: RuntimeManifest = validate_runtime_manifest(
        value,
        instances_sha256,
        corpus_sha256,
    )
    if manifest["answer_code_bundle_sha256"] != code_bundle_sha256_value:
        raise DownstreamDataError(
            "Answer code bundle hash mismatch: "
            f"expected={code_bundle_sha256_value}, "
            f"actual={manifest['answer_code_bundle_sha256']}"
        )
    return manifest


def selector_request_fingerprint(
    schema_version: str,
    instance_id: str,
    instance: Mapping[str, JsonLike],
    rendered_prompt: str,
    candidates: Sequence[CandidateDisplay],
    corpus_sha256: str,
    runtime_identity: Mapping[str, JsonLike],
    generation: SelectorGeneration,
    code_bundle_sha256_value: str,
) -> str:
    """Hash one complete selector execution request."""

    candidate_ids: list[str] = [candidate["skill_id"] for candidate in candidates]
    payload: dict[str, JsonLike] = {
        "schema_version": schema_version,
        "arm": "select",
        "instance_id": instance_id,
        "instance": instance,
        "instance_sha256": sha256_json(instance),
        "rendered_prompt": rendered_prompt,
        "ordered_candidate_skill_ids": candidate_ids,
        "candidate_displays": candidates,
        "corpus_sha256": corpus_sha256,
        "runtime_identity": runtime_identity,
        "generation": generation,
        "selector_code_bundle_sha256": code_bundle_sha256_value,
    }
    return sha256_json(payload)


def answer_execution_fingerprint(
    schema_version: str,
    arm: SemanticArm,
    instance_id: str,
    instance: Mapping[str, JsonLike],
    messages: Sequence[Mapping[str, JsonLike]],
    loaded_skills: Sequence[Mapping[str, JsonLike]],
    tools: Sequence[Mapping[str, JsonLike]],
    instances_sha256: str,
    corpus_sha256: str,
    runtime_identity: Mapping[str, JsonLike],
    generation: AnswerGeneration,
    code_bundle_sha256_value: str,
) -> str:
    """Hash one complete direct-answer execution request."""

    if arm not in SEMANTIC_ARMS:
        raise DownstreamDataError(f"Unknown semantic arm: arm={arm}")
    loaded_skill_ids: list[str] = []
    loaded_skill_content: list[dict[str, JsonLike]] = []
    for index, skill in enumerate(loaded_skills):
        skill_id: JsonLike | None = skill.get("skill_id")
        content: JsonLike | None = skill.get("content")
        if not isinstance(skill_id, str) or not skill_id:
            raise DownstreamDataError(
                "Loaded skill has invalid skill_id: "
                f"instance_id={instance_id}, index={index}, value={skill_id!r}"
            )
        if not isinstance(content, str):
            raise DownstreamDataError(
                "Loaded skill has invalid content: "
                f"instance_id={instance_id}, skill_id={skill_id}, "
                f"value_type={type(content).__name__}"
            )
        loaded_skill_ids.append(skill_id)
        loaded_skill_content.append(
            {
                "skill_id": skill_id,
                "content": content,
            }
        )
    payload: dict[str, JsonLike] = {
        "schema_version": schema_version,
        "arm": arm,
        "instance_id": instance_id,
        "instance": instance,
        "instance_sha256": sha256_json(instance),
        "question": instance.get("question"),
        "messages": messages,
        "loaded_skill_ids": loaded_skill_ids,
        "loaded_skills": loaded_skill_content,
        "tools": tools,
        "instances_sha256": instances_sha256,
        "corpus_sha256": corpus_sha256,
        "runtime_identity": runtime_identity,
        "generation": generation,
        "answer_code_bundle_sha256": code_bundle_sha256_value,
    }
    return sha256_json(payload)


def always_expected_skill_ids(
    retrieved: Sequence[Mapping[str, JsonLike]],
) -> tuple[str, ...]:
    """Return the mandatory routed top-1 skill for Always."""

    if not retrieved:
        raise DownstreamDataError("Always requires a non-empty retrieval list")
    skill_id: JsonLike | None = retrieved[0].get("skill_id")
    if not isinstance(skill_id, str) or not skill_id:
        raise DownstreamDataError(
            f"Always top-1 has invalid skill_id: value={skill_id!r}"
        )
    return (skill_id,)


def gated_expected_skill_ids(
    retrieved: Sequence[Mapping[str, JsonLike]],
) -> tuple[str, ...]:
    """Return zero or one routed skill after the gate."""

    if not retrieved:
        return ()
    skill_id: JsonLike | None = retrieved[0].get("skill_id")
    if not isinstance(skill_id, str) or not skill_id:
        raise DownstreamDataError(
            f"Gated top-1 has invalid skill_id: value={skill_id!r}"
        )
    return (skill_id,)


def select_expected_skill_ids(
    selected_skill_id: str | None,
    failure_category: FailureCategory,
) -> tuple[str, ...]:
    """Return the selector's expected loading decision."""

    if failure_category in ("success", "selector_fallback"):
        if not selected_skill_id:
            raise DownstreamDataError(
                "Successful selector record must contain selected_skill_id"
            )
        return (selected_skill_id,)
    if selected_skill_id is not None:
        raise DownstreamDataError(
            "Failed selector record must not contain selected_skill_id: "
            f"failure_category={failure_category}, selected_skill_id={selected_skill_id}"
        )
    return ()


def normalize_skill_ids(value: JsonLike | None, field_name: str) -> tuple[str, ...]:
    """Validate and normalize a JSON list of skill IDs."""

    if value is None:
        return ()
    if not isinstance(value, list):
        raise DownstreamDataError(
            "Skill IDs must be a JSON list: "
            f"field={field_name}, value_type={type(value).__name__}"
        )
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise DownstreamDataError(
                "Skill ID must be a non-empty string: "
                f"field={field_name}, index={index}, value={item!r}"
            )
        output.append(item)
    return tuple(output)


def audit_record_coverage(
    expected_instance_ids: Sequence[str],
    observed_instance_ids: Sequence[str],
) -> CoverageAudit:
    """Return missing, duplicate, and unexpected instance IDs."""

    expected_set: set[str] = set(expected_instance_ids)
    if len(expected_set) != len(expected_instance_ids):
        raise DownstreamDataError("Expected instance IDs contain duplicates")
    counts: dict[str, int] = {}
    for instance_id in observed_instance_ids:
        counts[instance_id] = counts.get(instance_id, 0) + 1
    observed_set: set[str] = set(counts)
    return CoverageAudit(
        expected=len(expected_instance_ids),
        observed=len(observed_instance_ids),
        missing_ids=tuple(sorted(expected_set - observed_set)),
        duplicate_ids=tuple(
            sorted(instance_id for instance_id, count in counts.items() if count > 1)
        ),
        unexpected_ids=tuple(sorted(observed_set - expected_set)),
    )


def validate_failure_category(value: JsonLike | None) -> FailureCategory:
    """Return one frozen failure category or raise."""

    if not isinstance(value, str) or value not in FAILURE_CATEGORIES:
        raise DownstreamDataError(
            f"Unknown failure category: value={value!r}, allowed={sorted(FAILURE_CATEGORIES)}"
        )
    return cast(FailureCategory, value)


def derive_legacy_answer_success(
    record: Mapping[str, JsonLike],
) -> FailureCategory:
    """Apply the frozen success-only adapter to one legacy answer record."""

    explicit_category: JsonLike | None = record.get("failure_category")
    if explicit_category is not None:
        category: FailureCategory = validate_failure_category(explicit_category)
        if category != "success":
            raise DownstreamDataError(
                "Legacy preseed only accepts successful answer records: "
                f"failure_category={category}"
            )
        return category
    error: JsonLike | None = record.get("error")
    raw_output: JsonLike | None = record.get("raw_output")
    if error not in (None, ""):
        raise DownstreamDataError(
            f"Legacy answer record contains an error: error={error!r}"
        )
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise DownstreamDataError(
            "Legacy answer record has no non-empty raw_output"
        )
    return "success"


def same_arm_preseed_eligibility(
    new_arm: SemanticArm,
    old_arm: SemanticArm,
    new_request_hash: str,
    old_request_hash: str,
    old_failure_category: FailureCategory,
    raw_output: JsonLike | None,
    old_skill_ids: Sequence[str],
    expected_skill_ids: Sequence[str],
    runtime_identity_matches: bool,
) -> PreseedEligibility:
    """Return whether a successful legacy answer may preseed the new arm."""

    if new_arm != old_arm:
        return PreseedEligibility(False, "semantic_arm_mismatch")
    if not runtime_identity_matches:
        return PreseedEligibility(False, "runtime_identity_mismatch")
    if new_request_hash != old_request_hash:
        return PreseedEligibility(False, "answer_request_hash_mismatch")
    if old_failure_category != "success":
        return PreseedEligibility(False, "legacy_record_not_success")
    if not isinstance(raw_output, str) or not raw_output.strip():
        return PreseedEligibility(False, "legacy_raw_output_empty")
    if tuple(old_skill_ids) != tuple(expected_skill_ids):
        return PreseedEligibility(False, "loaded_skill_ids_mismatch")
    return PreseedEligibility(True, "strict_same_arm_hash_match")


def classify_request_error(
    exception_name: str,
    message: str,
    status_code: int | None,
    response_body: str,
) -> FailureCategory:
    """Classify a request failure without hiding unknown causes."""

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
    transient_exception_names: frozenset[str] = frozenset(
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
    if exception_name in transient_exception_names or status_code in transient_statuses:
        return "infra_transient"
    return "unclassified_error"


def allowed_legacy_label(
    result_tag: str,
    arm: SemanticArm,
) -> str | None:
    """Return the only allowed legacy label for a model tag and semantic arm."""

    if result_tag not in RESULT_TAGS:
        raise DownstreamDataError(
            "Unknown result tag for legacy reuse: "
            f"result_tag={result_tag}, allowed={sorted(RESULT_TAGS)}"
        )
    if result_tag == "qwen3.5-4b-reference":
        qwen_labels: dict[SemanticArm, str | None] = {
            "routed_always": "always_r",
            "routed_gated": "gated_r",
            "routed_select": None,
            "fixed_gated": "gated",
        }
        return qwen_labels[arm]
    routed_select_label: str | None = (
        "select" if result_tag in SELECT_ELIGIBLE_FLEET_TAGS else None
    )
    fleet_labels: dict[SemanticArm, str | None] = {
        "routed_always": "always",
        "routed_gated": "gated",
        "routed_select": routed_select_label,
        "fixed_gated": None,
    }
    return fleet_labels[arm]
