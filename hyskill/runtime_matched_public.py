"""Build path-free public evidence from runtime-matched baseline records."""

from __future__ import annotations

import gzip
import io
from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

from hyskill.runtime_matched_execution import canonical_json


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

ANSWER_EVALUATION_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-evaluation-row-v1"
)
PUBLIC_ANSWER_SCHEMA_VERSION: str = (
    "runtime-matched-public-answer-row-v1"
)
PUBLIC_DECISION_SCHEMA_VERSION: str = (
    "runtime-matched-public-decision-row-v1"
)
PUBLIC_RUNTIME_JOB_SCHEMA_VERSION: str = (
    "runtime-matched-public-runtime-job-v1"
)
PUBLIC_USAGE_JOB_SCHEMA_VERSION: str = (
    "runtime-matched-public-usage-job-v1"
)
RERANK_DECISION_SCHEMA_VERSION: str = (
    "runtime-matched-rerank-decision-v1"
)
SELECT_DECISION_SCHEMA_VERSION: str = (
    "runtime-matched-select-decision-v1"
)
JOB_MANIFEST_SCHEMA_VERSION: str = "runtime-matched-job-manifest-v1"
RUNTIME_FACTS_SCHEMA_VERSION: str = "runtime-matched-runtime-facts-v1"
SHA256_LENGTH: int = 64
NATIVE_ARMS: tuple[str, ...] = ("always_rerank", "select_bm25")
RESOLVED_CATEGORIES: frozenset[str] = frozenset(
    {"success", "selector_fallback", "method_failure"}
)


class RuntimeMatchedPublicError(ValueError):
    """Raised when private evidence cannot form a valid public record."""


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return one JSON object."""

    if not isinstance(value, dict):
        raise RuntimeMatchedPublicError(
            f"Expected JSON object: context={context}, value={value!r}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return one JSON list."""

    if not isinstance(value, list):
        raise RuntimeMatchedPublicError(
            f"Expected JSON list: context={context}, value={value!r}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return one non-empty string."""

    if not isinstance(value, str) or not value:
        raise RuntimeMatchedPublicError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_optional_string(
    value: JsonValue | None,
    context: str,
) -> str | None:
    """Return one optional non-empty string."""

    if value is None:
        return None
    return require_string(value, context)


def require_boolean(value: JsonValue | None, context: str) -> bool:
    """Return one Boolean."""

    if not isinstance(value, bool):
        raise RuntimeMatchedPublicError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


def require_nonnegative_integer(
    value: JsonValue | None,
    context: str,
) -> int:
    """Return one nonnegative integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeMatchedPublicError(
            f"Expected nonnegative integer: context={context}, value={value!r}"
        )
    return value


def require_sha256(value: JsonValue | None, context: str) -> str:
    """Return one lowercase SHA-256 digest."""

    digest: str = require_string(value, context)
    if len(digest) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RuntimeMatchedPublicError(
            f"Expected SHA-256 digest: context={context}, value={digest!r}"
        )
    return digest


def require_string_list(
    value: JsonValue | None,
    context: str,
) -> list[str]:
    """Return a duplicate-free list of strings."""

    output: list[str] = [
        require_string(item, f"{context}[{index}]")
        for index, item in enumerate(require_list(value, context))
    ]
    if len(output) != len(set(output)):
        raise RuntimeMatchedPublicError(
            f"String list contains duplicates: context={context}"
        )
    return output


def require_identity(
    row: Mapping[str, JsonValue],
    expected: Mapping[str, JsonValue],
    context: str,
) -> None:
    """Require exact semantic identity fields."""

    mismatches: list[str] = [
        f"{field}:expected={value!r},actual={row.get(field)!r}"
        for field, value in expected.items()
        if row.get(field) != value
    ]
    if mismatches:
        raise RuntimeMatchedPublicError(
            f"Identity mismatch: context={context}, mismatches={mismatches}"
        )


def public_answer_row(
    row: Mapping[str, JsonValue],
    model: str,
    domain: str,
    arm: str,
) -> JsonObject:
    """Remove evaluator internals and retain reproducible answer evidence."""

    context: str = f"answer:{model}:{domain}:{arm}"
    require_identity(
        row,
        {
            "schema_version": ANSWER_EVALUATION_SCHEMA_VERSION,
            "model": model,
            "domain": domain,
            "arm": arm,
        },
        context,
    )
    category: str = require_string(
        row.get("failure_category"),
        f"{context}.failure_category",
    )
    if category not in {"success", "method_failure"}:
        raise RuntimeMatchedPublicError(
            f"Answer is unresolved: context={context}, category={category}"
        )
    skill_ids: list[str] = require_string_list(
        row.get("skill_ids_used"),
        f"{context}.skill_ids_used",
    )
    if arm == "bare" and skill_ids:
        raise RuntimeMatchedPublicError(
            f"Bare answer uses skills: context={context}, skills={skill_ids}"
        )
    return {
        "schema_version": PUBLIC_ANSWER_SCHEMA_VERSION,
        "model": model,
        "served_model": require_string(
            row.get("served_model"),
            f"{context}.served_model",
        ),
        "domain": domain,
        "arm": arm,
        "instance_id": require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        ),
        "correct": require_boolean(
            row.get("correct"),
            f"{context}.correct",
        ),
        "is_validation": require_boolean(
            row.get("is_validation"),
            f"{context}.is_validation",
        ),
        "failure_category": category,
        "skill_ids_used": skill_ids,
        "raw_output_sha256": require_sha256(
            row.get("raw_output_sha256"),
            f"{context}.raw_output_sha256",
        ),
        "answer_payload_hash": require_sha256(
            row.get("answer_payload_hash"),
            f"{context}.answer_payload_hash",
        ),
        "execution_request_hash": require_sha256(
            row.get("execution_request_hash"),
            f"{context}.execution_request_hash",
        ),
        "runtime_manifest_sha256": require_sha256(
            row.get("runtime_manifest_sha256"),
            f"{context}.runtime_manifest_sha256",
        ),
    }


def selected_original_rank(
    candidates: Sequence[str],
    selected_skill_id: str | None,
    context: str,
) -> int | None:
    """Return the selected skill's one-indexed BM25 rank."""

    if selected_skill_id is None:
        return None
    try:
        return list(candidates).index(selected_skill_id) + 1
    except ValueError as error:
        raise RuntimeMatchedPublicError(
            f"Selected skill is outside the candidate list: "
            f"context={context}, selected={selected_skill_id}"
        ) from error


def public_decision_row(
    row: Mapping[str, JsonValue],
    model: str,
    domain: str,
    arm: str,
) -> JsonObject:
    """Remove raw LLM text while preserving candidate and loading evidence."""

    if arm not in NATIVE_ARMS:
        raise ValueError(f"Unsupported native arm: arm={arm}")
    expected_schema: str = (
        RERANK_DECISION_SCHEMA_VERSION
        if arm == "always_rerank"
        else SELECT_DECISION_SCHEMA_VERSION
    )
    context: str = f"decision:{model}:{domain}:{arm}"
    require_identity(
        row,
        {
            "schema_version": expected_schema,
            "model": model,
            "domain": domain,
            "arm": arm,
            "stage": "decision",
        },
        context,
    )
    category: str = require_string(
        row.get("failure_category"),
        f"{context}.failure_category",
    )
    if category not in RESOLVED_CATEGORIES:
        raise RuntimeMatchedPublicError(
            f"Decision is unresolved: context={context}, category={category}"
        )
    candidates: list[str] = require_string_list(
        row.get("ordered_candidate_ids"),
        f"{context}.ordered_candidate_ids",
    )
    if len(candidates) != 50:
        raise RuntimeMatchedPublicError(
            f"Decision must bind 50 candidates: "
            f"context={context}, actual={len(candidates)}"
        )
    selected: str | None = require_optional_string(
        row.get("selected_skill_id"),
        f"{context}.selected_skill_id",
    )
    if category in {"success", "selector_fallback"} and selected is None:
        raise RuntimeMatchedPublicError(
            f"Resolved decision lacks a selected skill: context={context}"
        )
    if category == "method_failure" and selected is not None:
        raise RuntimeMatchedPublicError(
            f"Method failure selected a skill: context={context}"
        )
    source_field: str = (
        "source_sha256"
        if arm == "always_rerank"
        else "candidate_source_sha256"
    )
    payload_field: str = (
        "decision_payload_hash"
        if arm == "always_rerank"
        else "selector_payload_hash"
    )
    public_row: JsonObject = {
        "schema_version": PUBLIC_DECISION_SCHEMA_VERSION,
        "model": model,
        "served_model": require_string(
            row.get("served_model"),
            f"{context}.served_model",
        ),
        "domain": domain,
        "arm": arm,
        "instance_id": require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        ),
        "ordered_candidate_ids": candidates,
        "selected_skill_id": selected,
        "selected_original_rank": selected_original_rank(
            candidates,
            selected,
            context,
        ),
        "failure_category": category,
        "candidate_hash": require_sha256(
            row.get("candidate_hash"),
            f"{context}.candidate_hash",
        ),
        "candidate_source_sha256": require_sha256(
            row.get(source_field),
            f"{context}.{source_field}",
        ),
        "decision_payload_hash": require_sha256(
            row.get(payload_field),
            f"{context}.{payload_field}",
        ),
        "execution_request_hash": require_sha256(
            row.get("execution_request_hash"),
            f"{context}.execution_request_hash",
        ),
        "runtime_manifest_sha256": require_sha256(
            row.get("runtime_manifest_sha256"),
            f"{context}.runtime_manifest_sha256",
        ),
    }
    if arm == "always_rerank":
        reranked: list[str] = require_string_list(
            row.get("reranked_candidate_ids"),
            f"{context}.reranked_candidate_ids",
        )
        if category == "success" and (
            len(reranked) != 50 or reranked[0] != selected
        ):
            raise RuntimeMatchedPublicError(
                f"Successful rerank output is inconsistent: context={context}"
            )
        if category == "method_failure" and reranked:
            raise RuntimeMatchedPublicError(
                f"Failed rerank has a ranking: context={context}"
            )
        public_row["reranked_candidate_ids"] = reranked
        public_row["parse_attempts"] = require_nonnegative_integer(
            row.get("parse_attempts"),
            f"{context}.parse_attempts",
        )
    else:
        selected_rank: int | None = (
            None
            if row.get("selected_rank") is None
            else require_nonnegative_integer(
                row.get("selected_rank"),
                f"{context}.selected_rank",
            )
        )
        rank1_fallback: bool = require_boolean(
            row.get("rank1_fallback"),
            f"{context}.rank1_fallback",
        )
        if category in {"success", "selector_fallback"} and (
            selected_rank is None
            or selected_rank < 1
            or selected_rank > 50
            or candidates[selected_rank - 1] != selected
        ):
            raise RuntimeMatchedPublicError(
                f"Selector rank is inconsistent: context={context}"
            )
        if category == "selector_fallback" and (
            not rank1_fallback or selected_rank != 1
        ):
            raise RuntimeMatchedPublicError(
                f"Selector fallback is not rank 1: context={context}"
            )
        if category == "method_failure" and (
            selected_rank is not None or rank1_fallback
        ):
            raise RuntimeMatchedPublicError(
                f"Failed selector has a loading decision: context={context}"
            )
        public_row["selected_rank"] = selected_rank
        public_row["rank1_fallback"] = rank1_fallback
        public_row["parse_attempts"] = require_nonnegative_integer(
            row.get("parse_attempts"),
            f"{context}.parse_attempts",
        )
    return public_row


def public_artifacts(
    raw_artifacts: Sequence[JsonValue],
    context: str,
) -> list[JsonObject]:
    """Remove private artifact paths while retaining byte identity."""

    output: list[JsonObject] = []
    names: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact_context: str = f"{context}[{index}]"
        artifact: JsonObject = require_object(
            raw_artifact,
            artifact_context,
        )
        name: str = require_string(
            artifact.get("name"),
            f"{artifact_context}.name",
        )
        if name in names:
            raise RuntimeMatchedPublicError(
                f"Duplicate artifact name: context={context}, name={name}"
            )
        names.add(name)
        output.append(
            {
                "name": name,
                "size_bytes": require_nonnegative_integer(
                    artifact.get("size_bytes"),
                    f"{artifact_context}.size_bytes",
                ),
                "sha256": require_sha256(
                    artifact.get("sha256"),
                    f"{artifact_context}.sha256",
                ),
            }
        )
    return output


def public_code_files(
    raw_files: Sequence[JsonValue],
    context: str,
) -> list[JsonObject]:
    """Retain repository-relative code paths and immutable hashes."""

    output: list[JsonObject] = []
    paths: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        file_context: str = f"{context}[{index}]"
        code_file: JsonObject = require_object(raw_file, file_context)
        path: str = require_string(
            code_file.get("path"),
            f"{file_context}.path",
        )
        if path.startswith("/") or ".." in path.split("/"):
            raise RuntimeMatchedPublicError(
                f"Code path is not repository-relative: path={path!r}"
            )
        if path in paths:
            raise RuntimeMatchedPublicError(
                f"Duplicate code path: context={context}, path={path}"
            )
        paths.add(path)
        output.append(
            {
                "path": path,
                "size_bytes": require_nonnegative_integer(
                    code_file.get("size_bytes"),
                    f"{file_context}.size_bytes",
                ),
                "sha256": require_sha256(
                    code_file.get("sha256"),
                    f"{file_context}.sha256",
                ),
            }
        )
    return output


def public_runtime_job(
    manifest: Mapping[str, JsonValue],
    manifest_sha256: str,
) -> JsonObject:
    """Remove endpoint paths and hardware identifiers from one job manifest."""

    require_identity(
        manifest,
        {"schema_version": JOB_MANIFEST_SCHEMA_VERSION},
        "runtime-manifest",
    )
    facts: JsonObject = require_object(
        manifest.get("runtime_facts"),
        "runtime-manifest.runtime_facts",
    )
    require_identity(
        facts,
        {"schema_version": RUNTIME_FACTS_SCHEMA_VERSION},
        "runtime-facts",
    )
    job: JsonObject = require_object(facts.get("job"), "runtime-facts.job")
    checkpoint: JsonObject = require_object(
        facts.get("checkpoint"),
        "runtime-facts.checkpoint",
    )
    tokenizer: JsonObject = require_object(
        facts.get("tokenizer"),
        "runtime-facts.tokenizer",
    )
    endpoint: JsonObject = require_object(
        facts.get("endpoint"),
        "runtime-facts.endpoint",
    )
    software: JsonObject = require_object(
        facts.get("software"),
        "runtime-facts.software",
    )
    hardware: JsonObject = require_object(
        facts.get("hardware"),
        "runtime-facts.hardware",
    )
    source: JsonObject = require_object(
        facts.get("source"),
        "runtime-facts.source",
    )
    tokenizer_artifacts: JsonObject = require_object(
        tokenizer.get("artifacts"),
        "runtime-facts.tokenizer.artifacts",
    )
    public_tokenizer_artifacts: JsonObject = {
        name: require_sha256(
            digest,
            f"runtime-facts.tokenizer.artifacts.{name}",
        )
        for name, digest in sorted(tokenizer_artifacts.items())
    }
    raw_readback: JsonObject = require_object(
        endpoint.get("models_readback"),
        "runtime-facts.endpoint.models_readback",
    )
    readback_models: list[str] = [
        require_string(
            require_object(item, f"models-readback[{index}]").get("id"),
            f"models-readback[{index}].id",
        )
        for index, item in enumerate(
            require_list(raw_readback.get("data"), "models-readback.data")
        )
    ]
    return {
        "schema_version": PUBLIC_RUNTIME_JOB_SCHEMA_VERSION,
        "manifest_sha256": require_sha256(
            manifest_sha256,
            "runtime-manifest.sha256",
        ),
        "job": {
            field: require_string(job.get(field), f"runtime-facts.job.{field}")
            for field in (
                "job_id",
                "result_tag",
                "model",
                "domain",
                "arm",
            )
        },
        "checkpoint": {
            "repository": require_string(
                checkpoint.get("repository"),
                "runtime-facts.checkpoint.repository",
            ),
            "revision": require_string(
                checkpoint.get("revision"),
                "runtime-facts.checkpoint.revision",
            ),
            "provenance": require_string(
                checkpoint.get("provenance"),
                "runtime-facts.checkpoint.provenance",
            ),
            "files_manifest_sha256": require_sha256(
                checkpoint.get("files_manifest_sha256"),
                "runtime-facts.checkpoint.files_manifest_sha256",
            ),
        },
        "tokenizer": {
            "artifacts": public_tokenizer_artifacts,
            "chat_template_sha256": require_sha256(
                tokenizer.get("chat_template_sha256"),
                "runtime-facts.tokenizer.chat_template_sha256",
            ),
        },
        "endpoint": {
            "served_model": require_string(
                endpoint.get("served_model"),
                "runtime-facts.endpoint.served_model",
            ),
            "models_readback": readback_models,
            "vllm_version": require_string(
                endpoint.get("vllm_version"),
                "runtime-facts.endpoint.vllm_version",
            ),
            "dtype": require_string(
                endpoint.get("dtype"),
                "runtime-facts.endpoint.dtype",
            ),
            "quantization": require_string(
                endpoint.get("quantization"),
                "runtime-facts.endpoint.quantization",
            ),
            "max_model_len": require_nonnegative_integer(
                endpoint.get("max_model_len"),
                "runtime-facts.endpoint.max_model_len",
            ),
            "tensor_parallel_size": require_nonnegative_integer(
                endpoint.get("tensor_parallel_size"),
                "runtime-facts.endpoint.tensor_parallel_size",
            ),
        },
        "software": {
            name: require_string(
                software.get(name),
                f"runtime-facts.software.{name}",
            )
            for name in (
                "python_version",
                "pytorch_version",
                "transformers_version",
                "cuda_version",
                "driver_version",
            )
        },
        "hardware": {
            "gpu_model": require_string(
                hardware.get("gpu_model"),
                "runtime-facts.hardware.gpu_model",
            )
        },
        "source": {
            "sr_agents_revision": require_string(
                source.get("sr_agents_revision"),
                "runtime-facts.source.sr_agents_revision",
            )
        },
        "generation": cast(JsonValue, manifest.get("generation")),
        "artifacts": public_artifacts(
            require_list(manifest.get("artifacts"), "manifest.artifacts"),
            "manifest.artifacts",
        ),
        "code_files": public_code_files(
            require_list(manifest.get("code_files"), "manifest.code_files"),
            "manifest.code_files",
        ),
        "code_bundle_sha256": require_sha256(
            manifest.get("code_bundle_sha256"),
            "runtime-manifest.code_bundle_sha256",
        ),
    }


def public_usage_job(
    row: Mapping[str, JsonValue],
) -> JsonObject:
    """Remove one source-log path while retaining its measured usage."""

    identity: JsonObject = {
        field: require_string(row.get(field), f"usage-job.{field}")
        for field in ("model", "domain", "arm", "stage")
    }
    counts: JsonObject = {
        field: require_nonnegative_integer(
            row.get(field),
            f"usage-job.{field}",
        )
        for field in (
            "http_calls",
            "response_calls",
            "error_calls",
            "unique_instances",
            "usage_reported_calls",
            "usage_missing_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }
    return {
        "schema_version": PUBLIC_USAGE_JOB_SCHEMA_VERSION,
        **identity,
        **counts,
        "usage_missing_reasons": require_object(
            row.get("usage_missing_reasons"),
            "usage-job.usage_missing_reasons",
        ),
        "source_log_sha256": require_sha256(
            row.get("sha256"),
            "usage-job.sha256",
        ),
    }


def deterministic_jsonl_gzip(rows: Sequence[Mapping[str, JsonValue]]) -> bytes:
    """Return canonical JSONL gzip bytes with fixed metadata."""

    raw_bytes: bytes = "".join(
        f"{canonical_json(dict(row))}\n" for row in rows
    ).encode("utf-8")
    output: io.BytesIO = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=output,
        mtime=0,
    ) as gzip_file:
        gzip_file.write(raw_bytes)
    return output.getvalue()
