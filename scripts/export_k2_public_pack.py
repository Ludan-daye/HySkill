#!/usr/bin/env python3
"""Export one strictly validated, reproducible K=2 public result pack."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal, TypedDict, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    JsonLike,
    JsonObject,
    JsonValue,
    canonical_json,
    sha256_file,
    sha256_text,
)
from hyskill.k_ablation import (
    FIXED_VARIANTS,
    expected_router_pick,
    instance_ids_sha256,
    validation_ids,
)
from hyskill.loading_metrics import compute_loading_metrics


RETRIEVAL_DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
    "bigcodebench",
)
RULE_DOMAINS: tuple[str, ...] = RETRIEVAL_DOMAINS[:4]
ROUTED_ARMS: tuple[str, ...] = (
    "routed_always",
    "routed_gated",
    "routed_select",
)
ANSWER_ARM_ORDER: tuple[str, ...] = (
    "routed_always",
    "routed_gated",
    "routed_select",
    "fixed_gated",
)
SELECT_ELIGIBLE_MODELS: frozenset[str] = frozenset(
    {
        "qwen3.5-4b-reference",
        "qwen35-9b",
        "glm4-9b",
        "llama31-8b",
        "mistral7b",
    }
)
SELECT_INELIGIBLE_REASONS: dict[str, str] = {
    "deepseek7b": (
        "The frozen 50-candidate Select prompt exceeds this model's "
        "verified context support; the arm is unavailable, not zero."
    ),
    "yi15-9b": (
        "The frozen 50-candidate Select prompt exceeds this model's "
        "verified context support; the arm is unavailable, not zero."
    ),
}
SUPPORTED_MODELS: frozenset[str] = frozenset(
    {*SELECT_ELIGIBLE_MODELS, *SELECT_INELIGIBLE_REASONS}
)
FIXED_MODEL: str = "qwen3.5-4b-reference"
CACHE_MODEL_TAGS: dict[str, str] = {
    "qwen3.5-4b-reference": "qwen3.5-4b",
    "qwen35-9b": "qwen35-9b",
    "mistral7b": "mistral7b",
    "deepseek7b": "deepseek7b",
    "glm4-9b": "glm4-9b",
    "llama31-8b": "llama31-8b",
    "yi15-9b": "yi15-9b",
}
SHA256_LENGTH: int = 64
TOP_K: int = 50
GATE_P_MIN: float = 0.9
PUBLIC_PACK_SCHEMA: str = "k2-public-pack-v1"
EARLY_RAW_MODELS: frozenset[str] = frozenset(
    {"deepseek7b", "llama31-8b", "yi15-9b"}
)
PROVENANCE_LEVELS: frozenset[str] = frozenset(
    {
        "formal_direct",
        "posthoc_structural",
        "formal_retry_after_import",
    }
)

ArmName = Literal[
    "routed_always",
    "routed_gated",
    "routed_select",
    "fixed_gated",
]


class RetrievalData(TypedDict):
    """Validated routed retrieval evidence for one domain."""

    source_path: Path
    source_sha256: str
    ids: tuple[str, ...]
    gold_by_id: dict[str, list[str]]
    retrieved_by_id: dict[str, list[JsonObject]]
    validation_ids: frozenset[str]
    router: JsonObject
    public_rows: list[JsonObject]


class GateData(TypedDict):
    """Validated gate evidence for one rule-scored domain."""

    source_sha256: str
    expected_by_id: dict[str, list[str]]
    public_rows: list[JsonObject]


class FormalData(TypedDict):
    """Validated formal per-instance downstream evidence."""

    loading_rows: list[JsonObject]
    selection_rows: list[JsonObject]
    answer_rows: list[JsonObject]
    reuse_manifest: JsonObject
    formal_source_hashes: dict[str, str]
    formal_completion_evidence: JsonObject
    provenance_summary: JsonObject


class MetricData(TypedDict):
    """Validated per-model metric exports."""

    answer_metrics: JsonObject
    flat_rows: list[JsonObject]


class ProvenanceData(TypedDict):
    """Verified row labels and recovered raw-source line identities."""

    source_path: Path
    source_sha256: str
    rows: dict[tuple[str, str, str], JsonObject]
    source_line_hashes: dict[str, tuple[str, ...]]
    summary: JsonObject


def parse_args() -> argparse.Namespace:
    """Parse a fully explicit single-model export invocation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--formal-dir", required=True, type=Path)
    for domain in RETRIEVAL_DOMAINS:
        parser.add_argument(
            f"--{domain}-retrieval",
            required=True,
            type=Path,
        )
        parser.add_argument(
            f"--{domain}-router",
            required=True,
            type=Path,
        )
    for domain in RULE_DOMAINS:
        parser.add_argument(
            f"--{domain}-signals",
            required=True,
            type=Path,
        )
        parser.add_argument(
            f"--{domain}-taus",
            required=True,
            type=Path,
        )
        parser.add_argument(
            f"--{domain}-gated",
            required=True,
            type=Path,
        )
        parser.add_argument(
            f"--{domain}-fixed-gated",
            type=Path,
        )
    parser.add_argument("--loading-metrics-long", required=True, type=Path)
    parser.add_argument("--answer-metrics-long", required=True, type=Path)
    parser.add_argument("--answer-summary", required=True, type=Path)
    parser.add_argument("--significance", required=True, type=Path)
    parser.add_argument("--answer-provenance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return a JSON object or raise a contextual data error."""

    if not isinstance(value, dict):
        raise DownstreamDataError(
            f"Expected JSON object: context={context}, "
            f"actual={type(value).__name__}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return a JSON list or raise a contextual data error."""

    if not isinstance(value, list):
        raise DownstreamDataError(
            f"Expected JSON list: context={context}, "
            f"actual={type(value).__name__}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return a non-empty JSON string."""

    if not isinstance(value, str) or not value:
        raise DownstreamDataError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_boolean(value: JsonValue | None, context: str) -> bool:
    """Return a JSON Boolean."""

    if not isinstance(value, bool):
        raise DownstreamDataError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


def require_integer(value: JsonValue | None, context: str) -> int:
    """Return a JSON integer."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise DownstreamDataError(
            f"Expected integer: context={context}, value={value!r}"
        )
    return value


def require_number(value: JsonValue | None, context: str) -> float:
    """Return a finite JSON number."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise DownstreamDataError(
            f"Expected finite number: context={context}, value={value!r}"
        )
    return float(value)


def require_optional_number(
    value: JsonValue | None,
    context: str,
) -> float | None:
    """Return a finite number or an explicit null."""

    if value is None:
        return None
    return require_number(value, context)


def require_sha256(value: JsonValue | None, context: str) -> str:
    """Return one lowercase SHA-256 digest."""

    digest: str = require_string(value, context)
    if len(digest) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise DownstreamDataError(
            f"Expected lowercase SHA-256 digest: "
            f"context={context}, value={digest!r}"
        )
    return digest


def require_string_list(
    value: JsonValue | None,
    context: str,
) -> list[str]:
    """Return a duplicate-free list of non-empty strings."""

    raw_values: list[JsonValue] = require_list(value, context)
    values: list[str] = [
        require_string(item, f"{context}[{index}]")
        for index, item in enumerate(raw_values)
    ]
    if len(values) != len(set(values)):
        raise DownstreamDataError(
            f"String list contains duplicates: context={context}"
        )
    return values


def load_json(path: Path, context: str) -> JsonObject:
    """Load one required JSON object."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON input does not exist: context={context}, path={path}"
        )
    try:
        raw_value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise DownstreamDataError(
            "Malformed JSON input: "
            f"context={context}, path={path}, line={error.lineno}, "
            f"column={error.colno}, message={error.msg}"
        ) from error
    return require_object(raw_value, context)


def read_text_lines(path: Path, context: str) -> list[str]:
    """Read required plain or gzip-compressed UTF-8 lines."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSONL input does not exist: context={context}, path={path}"
        )
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as input_file:
            return input_file.readlines()
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load one required JSONL file and reject blank or malformed records."""

    rows: list[JsonObject] = []
    for line_number, line in enumerate(
        read_text_lines(path, context),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            raw_value: JsonValue = cast(JsonValue, json.loads(line))
        except json.JSONDecodeError as error:
            raise DownstreamDataError(
                "Malformed JSONL record: "
                f"context={context}, path={path}, line={line_number}, "
                f"column={error.colno}, message={error.msg}"
            ) from error
        rows.append(
            require_object(raw_value, f"{context}:{path}:{line_number}")
        )
    if not rows:
        raise DownstreamDataError(
            f"Required JSONL input is empty: context={context}, path={path}"
        )
    return rows


def index_rows(
    rows: Sequence[JsonObject],
    context: str,
) -> dict[str, JsonObject]:
    """Index rows by unique instance ID."""

    output: dict[str, JsonObject] = {}
    for row_index, row in enumerate(rows):
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}[{row_index}].instance_id",
        )
        if instance_id in output:
            raise DownstreamDataError(
                f"Duplicate instance ID: context={context}, "
                f"instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def verify_coverage(
    expected_ids: Sequence[str],
    observed_ids: Sequence[str],
    context: str,
) -> None:
    """Require exact, duplicate-free instance coverage."""

    if len(observed_ids) != len(set(observed_ids)):
        raise DownstreamDataError(
            f"Duplicate instance IDs in coverage: context={context}"
        )
    expected: set[str] = set(expected_ids)
    observed: set[str] = set(observed_ids)
    missing: list[str] = sorted(expected - observed)
    unexpected: list[str] = sorted(observed - expected)
    if missing or unexpected:
        raise DownstreamDataError(
            f"Instance coverage mismatch: context={context}, "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )


def argument_path(
    args: argparse.Namespace,
    domain: str,
    suffix: str,
) -> Path:
    """Resolve one required domain-specific CLI path."""

    raw_value: object = getattr(args, f"{domain}_{suffix}")
    if not isinstance(raw_value, Path):
        raise TypeError(
            f"CLI path was not parsed as Path: domain={domain}, suffix={suffix}"
        )
    return raw_value.resolve()


def fixed_source_paths(
    args: argparse.Namespace,
    model_tag: str,
) -> dict[str, Path]:
    """Return the exact explicit Qwen4 fixed-gated decision sources."""

    sources: dict[str, Path] = {}
    missing: list[str] = []
    for domain in RULE_DOMAINS:
        argument_name: str = f"{domain}_fixed_gated"
        raw_value: object = getattr(args, argument_name)
        if raw_value is None:
            if model_tag == FIXED_MODEL:
                missing.append(domain)
            continue
        if not isinstance(raw_value, Path):
            raise TypeError(
                "Fixed-gated CLI input is not a Path: "
                f"argument={argument_name}"
            )
        if model_tag != FIXED_MODEL:
            raise DownstreamDataError(
                "Fixed-gated CLI inputs are Qwen4-only: "
                f"model={model_tag}, argument={argument_name}"
            )
        sources[domain] = raw_value.resolve()
    if missing:
        raise FileNotFoundError(
            "Missing required Qwen4 fixed-gated CLI inputs: "
            f"domains={missing}"
        )
    return sources


def normalized_relative_path(value: JsonValue | None, context: str) -> str:
    """Return one normalized, non-parent relative POSIX path."""

    path_value: str = require_string(value, context)
    normalized: PurePosixPath = PurePosixPath(path_value)
    if (
        normalized.is_absolute()
        or ".." in normalized.parts
        or "\\" in path_value
        or str(normalized) != path_value
    ):
        raise DownstreamDataError(
            f"Expected normalized relative path: context={context}, "
            f"value={path_value!r}"
        )
    return path_value


def verified_raw_source_lines(
    manifest_path: Path,
    source_entries: Sequence[JsonValue],
) -> dict[str, tuple[str, ...]]:
    """Verify recovered raw files and index their stripped-line digests."""

    output: dict[str, tuple[str, ...]] = {}
    seen_paths: set[str] = set()
    for source_index, raw_entry in enumerate(source_entries):
        context: str = (
            f"answer-provenance.source_files[{source_index}]"
        )
        entry: JsonObject = require_object(raw_entry, context)
        relative_path: str = normalized_relative_path(
            entry.get("path"),
            f"{context}.path",
        )
        if relative_path in seen_paths:
            raise DownstreamDataError(
                f"Duplicate recovered raw-source path: path={relative_path}"
            )
        seen_paths.add(relative_path)
        source_path: Path = manifest_path.parent / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                "Recovered answer provenance source is missing: "
                f"path={source_path}"
            )
        expected_sha256: str = require_sha256(
            entry.get("sha256"),
            f"{context}.sha256",
        )
        actual_sha256: str = sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise DownstreamDataError(
                "Recovered raw-source hash mismatch: "
                f"path={source_path}, expected={expected_sha256}, "
                f"actual={actual_sha256}"
            )
        raw_lines: list[str] = read_text_lines(
            source_path,
            f"answer-provenance-source:{relative_path}",
        )
        nonblank_lines: list[str] = [
            line for line in raw_lines if line.strip()
        ]
        expected_rows: int = require_integer(
            entry.get("rows"),
            f"{context}.rows",
        )
        if expected_rows != len(nonblank_lines):
            raise DownstreamDataError(
                "Recovered raw-source row count mismatch: "
                f"path={source_path}, expected={expected_rows}, "
                f"actual={len(nonblank_lines)}"
            )
        line_hashes: list[str] = []
        for line_number, line in enumerate(nonblank_lines, start=1):
            try:
                raw_value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise DownstreamDataError(
                    "Recovered raw-source JSONL is malformed: "
                    f"path={source_path}, line={line_number}, "
                    f"column={error.colno}, message={error.msg}"
                ) from error
            require_object(
                raw_value,
                f"answer-provenance-source:{relative_path}:{line_number}",
            )
            line_hashes.append(sha256_text(line.rstrip("\r\n")))
        if actual_sha256 in output:
            raise DownstreamDataError(
                "Recovered raw sources duplicate a whole-file digest: "
                f"sha256={actual_sha256}"
            )
        output[actual_sha256] = tuple(line_hashes)
    return output


def load_answer_provenance(
    path: Path,
    model_tag: str,
) -> ProvenanceData:
    """Load an explicit, raw-source-verified answer provenance manifest."""

    payload: JsonObject = load_json(path, "answer-provenance")
    if payload.get("schema_version") != (
        "k2-answer-provenance-manifest-v1"
    ):
        raise DownstreamDataError(
            "Answer provenance schema mismatch: "
            f"actual={payload.get('schema_version')!r}"
        )
    if payload.get("model") != model_tag:
        raise DownstreamDataError(
            "Answer provenance model mismatch: "
            f"expected={model_tag}, actual={payload.get('model')!r}"
        )
    raw_sources_verified: bool = require_boolean(
        payload.get("raw_sources_verified"),
        "answer-provenance.raw_sources_verified",
    )
    raw_source_entries: list[JsonValue] = require_list(
        payload.get("source_files"),
        "answer-provenance.source_files",
    )
    if raw_source_entries and not raw_sources_verified:
        raise DownstreamDataError(
            "Recovered raw sources are present but not marked verified"
        )
    source_line_hashes: dict[str, tuple[str, ...]] = (
        verified_raw_source_lines(path, raw_source_entries)
        if raw_sources_verified
        else {}
    )
    raw_rows: list[JsonValue] = require_list(
        payload.get("rows"),
        "answer-provenance.rows",
    )
    if not raw_rows:
        raise DownstreamDataError(
            "Answer provenance manifest has no per-instance rows"
        )
    rows: dict[tuple[str, str, str], JsonObject] = {}
    observed_cohort_counts: dict[str, int] = {}
    observed_level_counts: dict[str, int] = {}
    for row_index, raw_row in enumerate(raw_rows):
        context: str = f"answer-provenance.rows[{row_index}]"
        row: JsonObject = require_object(raw_row, context)
        if row.get("schema_version") != "k2-answer-provenance-row-v1":
            raise DownstreamDataError(
                f"Answer provenance row schema mismatch: context={context}"
            )
        if row.get("model") != model_tag:
            raise DownstreamDataError(
                f"Answer provenance row model mismatch: context={context}, "
                f"expected={model_tag}, actual={row.get('model')!r}"
            )
        domain: str = require_string(
            row.get("domain"),
            f"{context}.domain",
        )
        if domain not in RULE_DOMAINS:
            raise DownstreamDataError(
                f"Unknown answer provenance domain: context={context}, "
                f"domain={domain}"
            )
        arm: str = require_string(row.get("arm"), f"{context}.arm")
        if arm not in expected_answer_arms(model_tag):
            raise DownstreamDataError(
                f"Unexpected answer provenance arm: context={context}, "
                f"arm={arm}"
            )
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        )
        level: str = require_string(
            row.get("provenance_level"),
            f"{context}.provenance_level",
        )
        if level not in PROVENANCE_LEVELS:
            raise DownstreamDataError(
                f"Unknown answer provenance level: context={context}, "
                f"level={level}"
            )
        cohort: str = require_string(
            row.get("cohort"),
            f"{context}.cohort",
        )
        key: tuple[str, str, str] = (domain, arm, instance_id)
        if key in rows:
            raise DownstreamDataError(
                f"Duplicate answer provenance row: key={key}"
            )
        rows[key] = row
        observed_cohort_counts[cohort] = (
            observed_cohort_counts.get(cohort, 0) + 1
        )
        observed_level_counts[level] = (
            observed_level_counts.get(level, 0) + 1
        )
    declared_cohort_counts: JsonObject = require_object(
        payload.get("cohort_counts"),
        "answer-provenance.cohort_counts",
    )
    declared_level_counts: JsonObject = require_object(
        payload.get("provenance_level_counts"),
        "answer-provenance.provenance_level_counts",
    )
    if canonical_json(declared_cohort_counts) != canonical_json(
        observed_cohort_counts
    ):
        raise DownstreamDataError(
            "Answer provenance cohort counts disagree with row labels: "
            f"declared={declared_cohort_counts}, "
            f"observed={observed_cohort_counts}"
        )
    if canonical_json(declared_level_counts) != canonical_json(
        observed_level_counts
    ):
        raise DownstreamDataError(
            "Answer provenance level counts disagree with row labels: "
            f"declared={declared_level_counts}, "
            f"observed={observed_level_counts}"
        )
    if (
        observed_level_counts.get("posthoc_structural", 0) > 0
        and not raw_sources_verified
    ):
        raise DownstreamDataError(
            "posthoc_structural rows require verified recovered raw sources"
        )
    summary: JsonObject = {
        "schema_version": "k2-public-answer-provenance-summary-v1",
        "model": model_tag,
        "rows": len(rows),
        "cohort_counts": observed_cohort_counts,
        "provenance_level_counts": observed_level_counts,
        "raw_sources_verified": raw_sources_verified,
        "source_files": [
            {
                "sha256": digest,
                "rows": len(line_hashes),
            }
            for digest, line_hashes in sorted(source_line_hashes.items())
        ],
        "manifest_sha256": sha256_file(path),
    }
    return {
        "source_path": path,
        "source_sha256": sha256_file(path),
        "rows": rows,
        "source_line_hashes": source_line_hashes,
        "summary": summary,
    }


def validate_k_stamp(
    metadata: JsonObject,
    model_tag: str,
    domain: str,
    expected_variant: str,
    context: str,
) -> None:
    """Validate the immutable K=2 model, domain, and variant identity."""

    stamp: JsonObject = require_object(
        metadata.get("k_ablation"),
        f"{context}.metadata.k_ablation",
    )
    expected_values: dict[str, JsonValue] = {
        "schema_version": 1,
        "tag": model_tag,
        "model": CACHE_MODEL_TAGS[model_tag],
        "k_samples": 2,
        "domain": domain,
        "variant": expected_variant,
    }
    for field, expected_value in expected_values.items():
        actual_value: JsonValue | None = stamp.get(field)
        if actual_value != expected_value:
            raise DownstreamDataError(
                "K=2 retrieval identity mismatch: "
                f"context={context}, field={field}, "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    cache: JsonObject = require_object(
        stamp.get("cache"),
        f"{context}.metadata.k_ablation.cache",
    )
    require_sha256(
        cache.get("manifest_sha256"),
        f"{context}.metadata.k_ablation.cache.manifest_sha256",
    )
    require_sha256(
        cache.get("artifact_sha256"),
        f"{context}.metadata.k_ablation.cache.artifact_sha256",
    )


def validate_router(
    value: JsonValue | None,
    instance_ids: frozenset[str],
    context: str,
) -> JsonObject:
    """Validate the frozen router decision and expose public fields."""

    router: JsonObject = require_object(value, context)
    if router.get("schema_version") != 1:
        raise DownstreamDataError(
            f"Router schema mismatch: context={context}, "
            f"actual={router.get('schema_version')!r}"
        )
    if router.get("validation_metric") != "nDCG@10":
        raise DownstreamDataError(
            f"Router metric mismatch: context={context}, "
            f"actual={router.get('validation_metric')!r}"
        )
    if require_number(
        router.get("validation_fraction"),
        f"{context}.validation_fraction",
    ) != 0.2:
        raise DownstreamDataError(
            f"Router validation fraction must be 0.2: context={context}"
        )
    if require_integer(router.get("seed"), f"{context}.seed") != 0:
        raise DownstreamDataError(
            f"Router seed must be 0: context={context}"
        )
    selected_validation_ids: frozenset[str] = validation_ids(instance_ids)
    if require_integer(
        router.get("n_validation"),
        f"{context}.n_validation",
    ) != len(selected_validation_ids):
        raise DownstreamDataError(
            f"Router validation count mismatch: context={context}"
        )
    expected_ids_sha256: str = instance_ids_sha256(selected_validation_ids)
    actual_ids_sha256: str = require_sha256(
        router.get("validation_ids_sha256"),
        f"{context}.validation_ids_sha256",
    )
    if actual_ids_sha256 != expected_ids_sha256:
        raise DownstreamDataError(
            "Router validation ID hash mismatch: "
            f"context={context}, expected={expected_ids_sha256}, "
            f"actual={actual_ids_sha256}"
        )
    raw_scores: JsonObject = require_object(
        router.get("validation_scores"),
        f"{context}.validation_scores",
    )
    if set(raw_scores) != set(FIXED_VARIANTS):
        raise DownstreamDataError(
            "Router scores must cover the frozen five variants: "
            f"context={context}, actual={sorted(raw_scores)}"
        )
    scores: dict[str, float] = {
        variant: require_number(
            raw_scores.get(variant),
            f"{context}.validation_scores.{variant}",
        )
        for variant in FIXED_VARIANTS
    }
    expected_pick, expected_degenerate = expected_router_pick(scores)
    pick: str = require_string(router.get("pick"), f"{context}.pick")
    if pick != expected_pick:
        raise DownstreamDataError(
            f"Router pick disagrees with scores: context={context}, "
            f"expected={expected_pick}, actual={pick}"
        )
    degenerate: bool = require_boolean(
        router.get("degenerate"),
        f"{context}.degenerate",
    )
    if degenerate != expected_degenerate:
        raise DownstreamDataError(
            f"Router degenerate flag mismatch: context={context}, "
            f"expected={expected_degenerate}, actual={degenerate}"
        )
    source_result: JsonObject = require_object(
        router.get("source_result"),
        f"{context}.source_result",
    )
    require_string(source_result.get("path"), f"{context}.source_result.path")
    source_result_sha256: str = require_sha256(
        source_result.get("sha256"),
        f"{context}.source_result.sha256",
    )
    return {
        "schema_version": "k2-public-router-domain-v1",
        "pick": pick,
        "validation_metric": "nDCG@10",
        "validation_scores": scores,
        "validation_ids": sorted(selected_validation_ids),
        "validation_ids_sha256": expected_ids_sha256,
        "validation_fraction": 0.2,
        "seed": 0,
        "degenerate": degenerate,
        "source_result_sha256": source_result_sha256,
    }


def validate_retrieval_payload(
    path: Path,
    model_tag: str,
    domain: str,
    context: str,
) -> RetrievalData:
    """Validate one complete K=2 routed top-50 source."""

    payload: JsonObject = load_json(path, context)
    metadata: JsonObject = require_object(
        payload.get("metadata"),
        f"{context}.metadata",
    )
    if metadata.get("dataset") != domain:
        raise DownstreamDataError(
            f"Retrieval domain mismatch: context={context}, "
            f"expected={domain}, actual={metadata.get('dataset')!r}"
        )
    if metadata.get("top_k") != TOP_K:
        raise DownstreamDataError(
            f"Retrieval top_k must be {TOP_K}: context={context}, "
            f"actual={metadata.get('top_k')!r}"
        )
    validate_k_stamp(metadata, model_tag, domain, "routed", context)
    raw_results: list[JsonValue] = require_list(
        payload.get("results"),
        f"{context}.results",
    )
    if not raw_results:
        raise DownstreamDataError(
            f"Retrieval source has no result rows: context={context}"
        )
    gold_by_id: dict[str, list[str]] = {}
    retrieved_by_id: dict[str, list[JsonObject]] = {}
    public_rows: list[JsonObject] = []
    source_sha256: str = sha256_file(path)
    for row_index, raw_result in enumerate(raw_results):
        row_context: str = f"{context}.results[{row_index}]"
        result: JsonObject = require_object(raw_result, row_context)
        instance_id: str = require_string(
            result.get("instance_id"),
            f"{row_context}.instance_id",
        )
        if instance_id in gold_by_id:
            raise DownstreamDataError(
                f"Duplicate retrieval instance ID: context={context}, "
                f"instance_id={instance_id}"
            )
        gold: list[str] = require_string_list(
            result.get("gold_skill_ids"),
            f"{row_context}.gold_skill_ids",
        )
        if not gold:
            raise DownstreamDataError(
                f"Retrieval row has no gold skill: context={row_context}"
            )
        raw_retrieved: list[JsonValue] = require_list(
            result.get("retrieved"),
            f"{row_context}.retrieved",
        )
        if len(raw_retrieved) != TOP_K:
            raise DownstreamDataError(
                f"Retrieval row must contain exactly {TOP_K} candidates: "
                f"context={row_context}, actual={len(raw_retrieved)}"
            )
        retrieved: list[JsonObject] = []
        skill_ids: list[str] = []
        for rank, raw_item in enumerate(raw_retrieved, start=1):
            item: JsonObject = require_object(
                raw_item,
                f"{row_context}.retrieved[{rank - 1}]",
            )
            skill_id: str = require_string(
                item.get("skill_id"),
                f"{row_context}.retrieved[{rank - 1}].skill_id",
            )
            score: float = require_number(
                item.get("score"),
                f"{row_context}.retrieved[{rank - 1}].score",
            )
            retrieved.append({"skill_id": skill_id, "score": score})
            skill_ids.append(skill_id)
        if len(skill_ids) != len(set(skill_ids)):
            raise DownstreamDataError(
                f"Retrieval candidates contain duplicate skills: "
                f"context={row_context}"
            )
        gold_by_id[instance_id] = gold
        retrieved_by_id[instance_id] = retrieved
        public_rows.append(
            {
                "schema_version": "k2-public-retrieval-row-v1",
                "model": model_tag,
                "domain": domain,
                "instance_id": instance_id,
                "k_samples": 2,
                "variant": "routed",
                "gold_skill_ids": gold,
                "retrieved": retrieved,
                "source_sha256": source_sha256,
            }
        )
    if metadata.get("n_queries") != len(raw_results):
        raise DownstreamDataError(
            f"Retrieval metadata count mismatch: context={context}, "
            f"expected={len(raw_results)}, actual={metadata.get('n_queries')!r}"
        )
    ids: tuple[str, ...] = tuple(sorted(gold_by_id))
    selected_validation_ids: frozenset[str] = validation_ids(frozenset(ids))
    router: JsonObject = validate_router(
        metadata.get("router"),
        frozenset(ids),
        f"{context}.metadata.router",
    )
    return {
        "source_path": path,
        "source_sha256": source_sha256,
        "ids": ids,
        "gold_by_id": gold_by_id,
        "retrieved_by_id": retrieved_by_id,
        "validation_ids": selected_validation_ids,
        "router": router,
        "public_rows": public_rows,
    }


def validate_router_source(
    path: Path,
    retrieval: RetrievalData,
    model_tag: str,
    domain: str,
) -> JsonObject:
    """Validate a separately supplied router source against retrieval."""

    context: str = f"router-source:{domain}"
    payload: JsonObject = load_json(path, context)
    metadata: JsonObject = require_object(
        payload.get("metadata"),
        f"{context}.metadata",
    )
    if metadata.get("dataset") != domain:
        raise DownstreamDataError(
            f"Router source domain mismatch: domain={domain}, "
            f"actual={metadata.get('dataset')!r}"
        )
    validate_k_stamp(metadata, model_tag, domain, "routed", context)
    raw_results: list[JsonValue] = require_list(
        payload.get("results"),
        f"{context}.results",
    )
    observed_ids: list[str] = [
        require_string(
            require_object(
                raw_result,
                f"{context}.results[{index}]",
            ).get("instance_id"),
            f"{context}.results[{index}].instance_id",
        )
        for index, raw_result in enumerate(raw_results)
    ]
    verify_coverage(retrieval["ids"], observed_ids, context)
    router: JsonObject = validate_router(
        metadata.get("router"),
        frozenset(retrieval["ids"]),
        f"{context}.metadata.router",
    )
    if canonical_json(router) != canonical_json(retrieval["router"]):
        raise DownstreamDataError(
            f"Router source disagrees with routed retrieval: domain={domain}"
        )
    router["router_source_sha256"] = sha256_file(path)
    router["retrieval_source_sha256"] = retrieval["source_sha256"]
    return router


def load_retrieval_sources(
    args: argparse.Namespace,
    model_tag: str,
) -> tuple[dict[str, RetrievalData], JsonObject, list[JsonObject]]:
    """Load all five retrieval and router sources."""

    retrievals: dict[str, RetrievalData] = {}
    router_domains: JsonObject = {}
    public_rows: list[JsonObject] = []
    for domain in RETRIEVAL_DOMAINS:
        retrieval_path: Path = argument_path(args, domain, "retrieval")
        router_path: Path = argument_path(args, domain, "router")
        retrieval: RetrievalData = validate_retrieval_payload(
            retrieval_path,
            model_tag,
            domain,
            f"retrieval:{domain}",
        )
        retrievals[domain] = retrieval
        router_domains[domain] = validate_router_source(
            router_path,
            retrieval,
            model_tag,
            domain,
        )
        public_rows.extend(retrieval["public_rows"])
    router_payload: JsonObject = {
        "schema_version": "k2-public-router-decisions-v1",
        "model": model_tag,
        "k_samples": 2,
        "domains": router_domains,
    }
    return retrievals, router_payload, public_rows


def validate_gate_sources(
    domain: str,
    model_tag: str,
    retrieval: RetrievalData,
    signals_path: Path,
    taus_path: Path,
    gated_path: Path,
) -> GateData:
    """Validate signals, calibration IDs, and gated decisions."""

    signals_payload: JsonObject = load_json(
        signals_path,
        f"gate-signals:{domain}",
    )
    if signals_payload.get("cache_misses") != 0:
        raise DownstreamDataError(
            f"Gate signals must have cache_misses=0: domain={domain}, "
            f"actual={signals_payload.get('cache_misses')!r}"
        )
    raw_signals: list[JsonValue] = require_list(
        signals_payload.get("signals"),
        f"gate-signals:{domain}.signals",
    )
    signal_rows: list[JsonObject] = [
        require_object(
            raw_signal,
            f"gate-signals:{domain}.signals[{index}]",
        )
        for index, raw_signal in enumerate(raw_signals)
    ]
    signal_index: dict[str, JsonObject] = index_rows(
        signal_rows,
        f"gate-signals:{domain}",
    )
    verify_coverage(
        retrieval["ids"],
        list(signal_index),
        f"gate-signals:{domain}",
    )

    taus: JsonObject = load_json(taus_path, f"gate-taus:{domain}")
    tau1: float | None = require_optional_number(
        taus.get("tau1"),
        f"gate-taus:{domain}.tau1",
    )
    tau2: float | None = require_optional_number(
        taus.get("tau2"),
        f"gate-taus:{domain}.tau2",
    )
    if require_number(
        taus.get("p_min"),
        f"gate-taus:{domain}.p_min",
    ) != GATE_P_MIN:
        raise DownstreamDataError(
            f"Gate p_min must be {GATE_P_MIN}: domain={domain}"
        )
    gate_validation_ids: list[str] = require_string_list(
        taus.get("val_ids"),
        f"gate-taus:{domain}.val_ids",
    )
    if frozenset(gate_validation_ids) != retrieval["validation_ids"]:
        raise DownstreamDataError(
            f"Gate validation IDs disagree with frozen split: domain={domain}"
        )
    if require_integer(
        taus.get("n_val"),
        f"gate-taus:{domain}.n_val",
    ) != len(gate_validation_ids):
        raise DownstreamDataError(
            f"Gate validation count mismatch: domain={domain}"
        )

    gated_payload: JsonObject = load_json(
        gated_path,
        f"gated-retrieval:{domain}",
    )
    gated_metadata: JsonObject = require_object(
        gated_payload.get("metadata"),
        f"gated-retrieval:{domain}.metadata",
    )
    if gated_metadata.get("dataset") != domain:
        raise DownstreamDataError(
            f"Gated source domain mismatch: domain={domain}"
        )
    validate_k_stamp(
        gated_metadata,
        model_tag,
        domain,
        "routed",
        f"gated-retrieval:{domain}",
    )
    raw_gated_results: list[JsonValue] = require_list(
        gated_payload.get("results"),
        f"gated-retrieval:{domain}.results",
    )
    gated_rows: list[JsonObject] = [
        require_object(
            raw_result,
            f"gated-retrieval:{domain}.results[{index}]",
        )
        for index, raw_result in enumerate(raw_gated_results)
    ]
    gated_index: dict[str, JsonObject] = index_rows(
        gated_rows,
        f"gated-retrieval:{domain}",
    )
    verify_coverage(
        retrieval["ids"],
        list(gated_index),
        f"gated-retrieval:{domain}",
    )

    public_rows: list[JsonObject] = []
    expected_by_id: dict[str, list[str]] = {}
    decision_counts: dict[str, int] = {
        "blocked_s1": 0,
        "skipped_s2": 0,
        "kept": 0,
    }
    signals_sha256: str = sha256_file(signals_path)
    taus_sha256: str = sha256_file(taus_path)
    gated_sha256: str = sha256_file(gated_path)
    for instance_id in retrieval["ids"]:
        signal: JsonObject = signal_index[instance_id]
        top1: str = require_string(
            signal.get("top1"),
            f"gate-signals:{domain}:{instance_id}.top1",
        )
        expected_top1: str = require_string(
            retrieval["retrieved_by_id"][instance_id][0].get("skill_id"),
            f"retrieval:{domain}:{instance_id}.top1",
        )
        if top1 != expected_top1:
            raise DownstreamDataError(
                f"Gate signal top-1 mismatch: domain={domain}, "
                f"instance_id={instance_id}, expected={expected_top1}, "
                f"actual={top1}"
            )
        s1: float = require_number(
            signal.get("S1"),
            f"gate-signals:{domain}:{instance_id}.S1",
        )
        s2: float = require_number(
            signal.get("S2"),
            f"gate-signals:{domain}:{instance_id}.S2",
        )
        truth_wrong: bool = require_boolean(
            signal.get("rel_truth_wrong"),
            f"gate-signals:{domain}:{instance_id}.rel_truth_wrong",
        )
        if truth_wrong != (
            top1 not in set(retrieval["gold_by_id"][instance_id])
        ):
            raise DownstreamDataError(
                f"Gate relevance label mismatch: domain={domain}, "
                f"instance_id={instance_id}"
            )
        if tau1 is not None and s1 < tau1:
            decision: str = "blocked_s1"
        elif tau2 is not None and s2 < tau2:
            decision = "skipped_s2"
        else:
            decision = "kept"
        raw_gated: list[JsonValue] = require_list(
            gated_index[instance_id].get("retrieved"),
            f"gated-retrieval:{domain}:{instance_id}.retrieved",
        )
        gated_retrieved: list[JsonObject] = [
            require_object(
                raw_item,
                f"gated-retrieval:{domain}:{instance_id}.retrieved[{index}]",
            )
            for index, raw_item in enumerate(raw_gated)
        ]
        if decision == "kept":
            if canonical_json(gated_retrieved) != canonical_json(
                retrieval["retrieved_by_id"][instance_id]
            ):
                raise DownstreamDataError(
                    "Gate changed routed candidates instead of preserving "
                    f"the source: domain={domain}, instance_id={instance_id}"
                )
            expected: list[str] = [top1]
        else:
            if gated_retrieved:
                raise DownstreamDataError(
                    f"Gate should have cleared the routed result: "
                    f"domain={domain}, instance_id={instance_id}, "
                    f"decision={decision}"
                )
            expected = []
        expected_by_id[instance_id] = expected
        decision_counts[decision] += 1
        public_rows.append(
            {
                "schema_version": "k2-public-gating-row-v1",
                "model": model_tag,
                "domain": domain,
                "instance_id": instance_id,
                "top1_skill_id": top1,
                "gold_skill_ids": retrieval["gold_by_id"][instance_id],
                "S1": s1,
                "S2": s2,
                "tau1": tau1,
                "tau2": tau2,
                "decision": decision,
                "loaded": bool(expected),
                "is_validation": (
                    instance_id in retrieval["validation_ids"]
                ),
                "signals_source_sha256": signals_sha256,
                "taus_source_sha256": taus_sha256,
                "gated_source_sha256": gated_sha256,
            }
        )
    gate_metadata: JsonObject = require_object(
        gated_metadata.get("gate"),
        f"gated-retrieval:{domain}.metadata.gate",
    )
    expected_gate_values: dict[str, JsonValue] = {
        "tau1": tau1,
        "tau2": tau2,
        "blocked": decision_counts["blocked_s1"],
        "skipped": decision_counts["skipped_s2"],
        "kept": decision_counts["kept"],
    }
    for field, expected_value in expected_gate_values.items():
        if gate_metadata.get(field) != expected_value:
            raise DownstreamDataError(
                "Gated metadata mismatch: "
                f"domain={domain}, field={field}, "
                f"expected={expected_value!r}, "
                f"actual={gate_metadata.get(field)!r}"
            )
    return {
        "source_sha256": gated_sha256,
        "expected_by_id": expected_by_id,
        "public_rows": public_rows,
    }


def load_gate_sources(
    args: argparse.Namespace,
    model_tag: str,
    retrievals: Mapping[str, RetrievalData],
) -> tuple[dict[str, GateData], list[JsonObject]]:
    """Load all four rule-domain gate evidence sets."""

    gates: dict[str, GateData] = {}
    public_rows: list[JsonObject] = []
    for domain in RULE_DOMAINS:
        gate: GateData = validate_gate_sources(
            domain,
            model_tag,
            retrievals[domain],
            argument_path(args, domain, "signals"),
            argument_path(args, domain, "taus"),
            argument_path(args, domain, "gated"),
        )
        gates[domain] = gate
        public_rows.extend(gate["public_rows"])
    return gates, public_rows


def expected_answer_arms(model_tag: str) -> tuple[ArmName, ...]:
    """Return the exact active answer arms for one model."""

    arms: list[ArmName] = ["routed_always", "routed_gated"]
    if model_tag in SELECT_ELIGIBLE_MODELS:
        arms.append("routed_select")
    if model_tag == FIXED_MODEL:
        arms.append("fixed_gated")
    return tuple(arms)


def expected_formal_files(model_tag: str) -> set[str]:
    """Return the complete required formal root file set."""

    names: set[str] = set()
    for domain in RULE_DOMAINS:
        names.update(
            {
                f"{domain}-routed-always-gated.loading.jsonl",
                f"{domain}-routed-always.jsonl",
                f"{domain}-routed-always.eval.json",
                f"{domain}-routed-gated.jsonl",
                f"{domain}-routed-gated.eval.json",
            }
        )
        if model_tag in SELECT_ELIGIBLE_MODELS:
            names.update(
                {
                    f"{domain}-routed-select.selection.jsonl",
                    f"{domain}-routed-select-source.json",
                    f"{domain}-routed-select.loading.jsonl",
                    f"{domain}-routed-select.jsonl",
                    f"{domain}-routed-select.eval.json",
                }
            )
        if model_tag == FIXED_MODEL:
            names.update(
                {
                    f"{domain}-fixed-gated.jsonl",
                    f"{domain}-fixed-gated.eval.json",
                }
            )
    return names


def known_optional_formal_files(model_tag: str) -> set[str]:
    """Return plan-defined formal files that are not exporter requirements."""

    names: set[str] = {"FORMAL_COMPLETE", "manifest.json"}
    for domain in RULE_DOMAINS:
        names.update(
            {
                f"{domain}-routed-signals.json",
                f"{domain}-routed-taus.json",
                f"{domain}-routed-gated.json",
            }
        )
        if model_tag == FIXED_MODEL:
            names.update(
                {
                    f"{domain}-fixed-gated.json",
                    f"{domain}-fixed-signals.json",
                    f"{domain}-fixed-taus.json",
                }
            )
    return names


def validate_formal_complete_marker(
    formal_dir: Path,
    model_tag: str,
) -> JsonObject:
    """Validate the required model completion marker and its producer count."""

    marker_path: Path = formal_dir / "FORMAL_COMPLETE"
    if not marker_path.exists():
        raise FileNotFoundError(
            f"Required FORMAL_COMPLETE marker is missing: path={marker_path}"
        )
    if not marker_path.is_file():
        raise DownstreamDataError(
            f"FORMAL_COMPLETE must be a regular file: path={marker_path}"
        )
    content: str = marker_path.read_text(encoding="utf-8").strip()
    parts: list[str] = content.split()
    expected_prefix: str = (
        "K2_ELIGIBLE_MODEL_FORMAL_COMPLETE"
        if model_tag in SELECT_ELIGIBLE_MODELS
        else "K2_INELIGIBLE_MODEL_FORMAL_COMPLETE"
    )
    if not parts or parts[0] != expected_prefix:
        raise DownstreamDataError(
            f"Malformed FORMAL_COMPLETE marker: path={marker_path}, "
            f"expected_prefix={expected_prefix!r}, content={content!r}"
        )
    fields: dict[str, str] = {}
    for raw_field in parts[1:]:
        name, separator, value = raw_field.partition("=")
        if separator != "=" or not name or not value or name in fields:
            raise DownstreamDataError(
                f"Malformed FORMAL_COMPLETE field: path={marker_path}, "
                f"field={raw_field!r}"
            )
        fields[name] = value
    if set(fields) != {"result_tag", "completions"}:
        raise DownstreamDataError(
            f"FORMAL_COMPLETE fields mismatch: path={marker_path}, "
            f"actual={sorted(fields)}"
        )
    answer_completions: int = (
        len(expected_answer_arms(model_tag)) * len(RULE_DOMAINS)
    )
    selection_completions: int = (
        len(RULE_DOMAINS)
        if model_tag in SELECT_ELIGIBLE_MODELS
        else 0
    )
    expected_completions: int = answer_completions + selection_completions
    if fields["result_tag"] != model_tag:
        raise DownstreamDataError(
            f"FORMAL_COMPLETE model mismatch: expected={model_tag}, "
            f"actual={fields['result_tag']}"
        )
    if (
        not fields["completions"].isdigit()
        or int(fields["completions"]) != expected_completions
    ):
        raise DownstreamDataError(
            "FORMAL_COMPLETE completion count mismatch: "
            f"expected={expected_completions}, "
            f"actual={fields['completions']!r}"
        )
    return {
        "present": True,
        "sha256": sha256_file(marker_path),
        "declared_completions": expected_completions,
    }


def validate_formal_root(
    formal_dir: Path,
    model_tag: str,
) -> JsonObject:
    """Reject incomplete formal roots and every unknown root entry."""

    if not formal_dir.is_dir():
        raise FileNotFoundError(
            f"Formal result directory does not exist: path={formal_dir}"
        )
    expected: set[str] = expected_formal_files(model_tag)
    allowed: set[str] = {
        *expected,
        *known_optional_formal_files(model_tag),
        "audits",
        "logs",
    }
    actual: set[str] = {entry.name for entry in formal_dir.iterdir()}
    unknown: list[str] = sorted(actual - allowed)
    if unknown:
        raise DownstreamDataError(
            f"Formal result root contains unknown entries: entries={unknown}"
        )
    missing: list[str] = sorted(expected - actual)
    if missing:
        raise FileNotFoundError(
            f"Formal result root is incomplete: missing={missing}"
        )
    if not (formal_dir / "audits").is_dir():
        raise FileNotFoundError(
            f"Formal audits directory is missing: path={formal_dir / 'audits'}"
        )
    for filename in expected:
        path: Path = formal_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Formal result entry is not a regular file: path={path}"
            )
    marker_path: Path = formal_dir / "FORMAL_COMPLETE"
    if marker_path.exists():
        return validate_formal_complete_marker(formal_dir, model_tag)
    return {
        "present": False,
        "sha256": None,
        "declared_completions": None,
    }


def expected_routed_skill_ids(
    arm: str,
    instance_id: str,
    retrieval: RetrievalData,
    gate: GateData,
    selected_by_id: Mapping[str, list[str]],
) -> list[str]:
    """Return the frozen decision for one routed arm."""

    if arm == "routed_always":
        return [
            require_string(
                retrieval["retrieved_by_id"][instance_id][0].get("skill_id"),
                f"retrieval:{instance_id}.top1",
            )
        ]
    if arm == "routed_gated":
        return list(gate["expected_by_id"][instance_id])
    if arm == "routed_select":
        if instance_id not in selected_by_id:
            raise DownstreamDataError(
                f"Select decision is missing: instance_id={instance_id}"
            )
        return list(selected_by_id[instance_id])
    raise DownstreamDataError(f"Unknown routed arm: arm={arm}")


def validate_loading_row(
    row: JsonObject,
    model_tag: str,
    domain: str,
    arm: str,
    expected_skill_ids: list[str],
    gold_skill_ids: list[str],
    is_validation: bool,
    expected_source_sha256: str,
    context: str,
) -> JsonObject:
    """Validate one decision-level loading record."""

    if row.get("schema_version") != "k2-loading-decision-v1":
        raise DownstreamDataError(
            f"Loading schema mismatch: context={context}, "
            f"actual={row.get('schema_version')!r}"
        )
    expected_identity: dict[str, JsonValue] = {
        "model": model_tag,
        "domain": domain,
        "arm": arm,
        "expected_skill_ids": expected_skill_ids,
        "gold_skill_ids": gold_skill_ids,
        "loaded": bool(expected_skill_ids),
        "is_validation": is_validation,
        "decision_source_sha256": expected_source_sha256,
    }
    for field, expected_value in expected_identity.items():
        if row.get(field) != expected_value:
            raise DownstreamDataError(
                f"Loading row mismatch: context={context}, field={field}, "
                f"expected={expected_value!r}, actual={row.get(field)!r}"
            )
    hit: bool | None = (
        any(skill_id in set(gold_skill_ids) for skill_id in expected_skill_ids)
        if expected_skill_ids
        else None
    )
    if row.get("hit") != hit or row.get("gold_loaded") != (hit is True):
        raise DownstreamDataError(
            f"Loading hit fields are inconsistent: context={context}"
        )
    failure_category: str = require_string(
        row.get("failure_category"),
        f"{context}.failure_category",
    )
    if arm != "routed_select" and failure_category != "success":
        raise DownstreamDataError(
            f"Deterministic loading arm cannot fail: context={context}, "
            f"failure_category={failure_category}"
        )
    return {
        "schema_version": "k2-loading-decision-v1",
        "instance_id": require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        ),
        "model": model_tag,
        "domain": domain,
        "arm": arm,
        "expected_skill_ids": expected_skill_ids,
        "gold_skill_ids": gold_skill_ids,
        "loaded": bool(expected_skill_ids),
        "hit": hit,
        "gold_loaded": hit is True,
        "is_validation": is_validation,
        "failure_category": failure_category,
        "decision_source_sha256": expected_source_sha256,
    }


def validate_selection_completion(
    formal_dir: Path,
    model_tag: str,
    domain: str,
    selection_path: Path,
    selected_source_path: Path,
    selection_rows: Sequence[JsonObject],
) -> str:
    """Validate one producer-issued selection completion artifact."""

    completion_path: Path = (
        formal_dir
        / "audits"
        / f"{domain}-routed-select.selection-completion.json"
    )
    attempt_log_path: Path = (
        formal_dir / "logs" / f"{domain}-routed-select.attempts.jsonl"
    )
    completion: JsonObject = load_json(
        completion_path,
        f"selection-completion:{domain}",
    )
    attempt_rows: list[JsonObject] = load_jsonl(
        attempt_log_path,
        f"selection-attempts:{domain}",
    )
    failure_categories: dict[str, int] = {}
    for index, row in enumerate(selection_rows):
        category: str = require_string(
            row.get("failure_category"),
            f"selection-completion:{domain}.selection[{index}]"
            ".failure_category",
        )
        failure_categories[category] = (
            failure_categories.get(category, 0) + 1
        )
    expected_fields: dict[str, JsonValue] = {
        "schema_version": "k2-selection-validation-v1",
        "valid": True,
        "model": CACHE_MODEL_TAGS[model_tag],
        "domain": domain,
        "expected": len(selection_rows),
        "observed": len(selection_rows),
        "failure_categories": failure_categories,
        "attempt_records": len(attempt_rows),
        "selection_sha256": sha256_file(selection_path),
        "selected_source_sha256": sha256_file(selected_source_path),
        "attempt_log_sha256": sha256_file(attempt_log_path),
    }
    for field, expected_value in expected_fields.items():
        if completion.get(field) != expected_value:
            raise DownstreamDataError(
                "Selection completion mismatch: "
                f"domain={domain}, field={field}, "
                f"expected={expected_value!r}, "
                f"actual={completion.get(field)!r}"
            )
    return sha256_file(completion_path)


def validate_selection_domain(
    formal_dir: Path,
    model_tag: str,
    domain: str,
    retrieval: RetrievalData,
) -> tuple[list[JsonObject], dict[str, list[str]], str, str]:
    """Validate selection records and selected-source decisions."""

    selection_path: Path = (
        formal_dir / f"{domain}-routed-select.selection.jsonl"
    )
    selected_source_path: Path = (
        formal_dir / f"{domain}-routed-select-source.json"
    )
    selection_rows: list[JsonObject] = load_jsonl(
        selection_path,
        f"selection:{domain}",
    )
    selection_index: dict[str, JsonObject] = index_rows(
        selection_rows,
        f"selection:{domain}",
    )
    verify_coverage(
        retrieval["ids"],
        list(selection_index),
        f"selection:{domain}",
    )
    selected_payload: JsonObject = load_json(
        selected_source_path,
        f"selected-source:{domain}",
    )
    raw_selected_results: list[JsonValue] = require_list(
        selected_payload.get("results"),
        f"selected-source:{domain}.results",
    )
    selected_rows: list[JsonObject] = [
        require_object(
            raw_row,
            f"selected-source:{domain}.results[{index}]",
        )
        for index, raw_row in enumerate(raw_selected_results)
    ]
    selected_index: dict[str, JsonObject] = index_rows(
        selected_rows,
        f"selected-source:{domain}",
    )
    verify_coverage(
        retrieval["ids"],
        list(selected_index),
        f"selected-source:{domain}",
    )
    selected_by_id: dict[str, list[str]] = {}
    public_rows: list[JsonObject] = []
    retrieval_source_sha256: str = retrieval["source_sha256"]
    for instance_id in retrieval["ids"]:
        row: JsonObject = selection_index[instance_id]
        context: str = f"selection:{domain}:{instance_id}"
        if row.get("schema_version") != "k2-selection-record-v1":
            raise DownstreamDataError(
                f"Selection schema mismatch: context={context}"
            )
        expected_identity: dict[str, JsonValue] = {
            "instance_id": instance_id,
            "dataset": domain,
            "arm": "routed_select",
            "model": CACHE_MODEL_TAGS[model_tag],
            "source_sha256": retrieval_source_sha256,
        }
        for field, expected_value in expected_identity.items():
            if row.get(field) != expected_value:
                raise DownstreamDataError(
                    f"Selection identity mismatch: context={context}, "
                    f"field={field}, expected={expected_value!r}, "
                    f"actual={row.get(field)!r}"
                )
        ordered_candidate_ids: list[str] = require_string_list(
            row.get("ordered_candidate_ids"),
            f"{context}.ordered_candidate_ids",
        )
        expected_candidates: list[str] = [
            require_string(
                item.get("skill_id"),
                f"retrieval:{domain}:{instance_id}.skill_id",
            )
            for item in retrieval["retrieved_by_id"][instance_id]
        ]
        if ordered_candidate_ids != expected_candidates:
            raise DownstreamDataError(
                f"Selection candidate order mismatch: context={context}"
            )
        if len(ordered_candidate_ids) != TOP_K:
            raise DownstreamDataError(
                f"Selection candidate count must be {TOP_K}: context={context}"
            )
        selected_skill_id: str = require_string(
            row.get("selected_skill_id"),
            f"{context}.selected_skill_id",
        )
        selected_rank: int = require_integer(
            row.get("selected_rank"),
            f"{context}.selected_rank",
        )
        if (
            selected_rank < 1
            or selected_rank > TOP_K
            or ordered_candidate_ids[selected_rank - 1] != selected_skill_id
        ):
            raise DownstreamDataError(
                f"Selection rank does not identify the chosen skill: "
                f"context={context}, rank={selected_rank}, "
                f"skill_id={selected_skill_id}"
            )
        rank1_fallback: bool = require_boolean(
            row.get("rank1_fallback"),
            f"{context}.rank1_fallback",
        )
        failure_category: str = require_string(
            row.get("failure_category"),
            f"{context}.failure_category",
        )
        if failure_category not in {"success", "selector_fallback"}:
            raise DownstreamDataError(
                f"Selection contains unresolved failure: context={context}, "
                f"failure_category={failure_category}"
            )
        if failure_category == "selector_fallback" and (
            not rank1_fallback or selected_rank != 1
        ):
            raise DownstreamDataError(
                f"Selector fallback must select rank 1: context={context}"
            )
        raw_source_retrieved: list[JsonValue] = require_list(
            selected_index[instance_id].get("retrieved"),
            f"selected-source:{domain}:{instance_id}.retrieved",
        )
        source_skill_ids: list[str] = [
            require_string(
                require_object(
                    raw_item,
                    f"selected-source:{domain}:{instance_id}"
                    f".retrieved[{index}]",
                ).get("skill_id"),
                f"selected-source:{domain}:{instance_id}"
                f".retrieved[{index}].skill_id",
            )
            for index, raw_item in enumerate(raw_source_retrieved)
        ]
        if source_skill_ids != [selected_skill_id]:
            raise DownstreamDataError(
                f"Selected source disagrees with selection: context={context}"
            )
        selected_by_id[instance_id] = [selected_skill_id]
        public_rows.append(
            {
                "schema_version": "k2-public-selection-row-v1",
                "model": model_tag,
                "domain": domain,
                "instance_id": instance_id,
                "ordered_candidate_ids": ordered_candidate_ids,
                "candidate_hash": require_sha256(
                    row.get("candidate_hash"),
                    f"{context}.candidate_hash",
                ),
                "selector_request_hash": require_sha256(
                    row.get("selector_request_hash"),
                    f"{context}.selector_request_hash",
                ),
                "selected_skill_id": selected_skill_id,
                "selected_rank": selected_rank,
                "rank1_fallback": rank1_fallback,
                "failure_category": failure_category,
                "source_sha256": retrieval_source_sha256,
            }
        )
    completion_sha256: str = validate_selection_completion(
        formal_dir,
        model_tag,
        domain,
        selection_path,
        selected_source_path,
        selection_rows,
    )
    return (
        public_rows,
        selected_by_id,
        sha256_file(selected_source_path),
        completion_sha256,
    )


def validate_fixed_source(
    source_path: Path,
    model_tag: str,
    domain: str,
    retrieval: RetrievalData,
) -> tuple[dict[str, list[str]], str]:
    """Validate the Qwen4 fixed-gated answer decision source."""

    payload: JsonObject = load_json(
        source_path,
        f"fixed-gated:{domain}",
    )
    metadata: JsonObject = require_object(
        payload.get("metadata"),
        f"fixed-gated:{domain}.metadata",
    )
    if metadata.get("dataset") != domain:
        raise DownstreamDataError(
            f"Fixed-gated domain mismatch: domain={domain}"
        )
    validate_k_stamp(
        metadata,
        model_tag,
        domain,
        "naive_skill",
        f"fixed-gated:{domain}",
    )
    raw_results: list[JsonValue] = require_list(
        payload.get("results"),
        f"fixed-gated:{domain}.results",
    )
    rows: list[JsonObject] = [
        require_object(
            raw_row,
            f"fixed-gated:{domain}.results[{index}]",
        )
        for index, raw_row in enumerate(raw_results)
    ]
    index: dict[str, JsonObject] = index_rows(rows, f"fixed-gated:{domain}")
    verify_coverage(
        retrieval["ids"],
        list(index),
        f"fixed-gated:{domain}",
    )
    expected_by_id: dict[str, list[str]] = {}
    for instance_id in retrieval["ids"]:
        raw_retrieved: list[JsonValue] = require_list(
            index[instance_id].get("retrieved"),
            f"fixed-gated:{domain}:{instance_id}.retrieved",
        )
        if raw_retrieved:
            first: JsonObject = require_object(
                raw_retrieved[0],
                f"fixed-gated:{domain}:{instance_id}.retrieved[0]",
            )
            expected_by_id[instance_id] = [
                require_string(
                    first.get("skill_id"),
                    f"fixed-gated:{domain}:{instance_id}"
                    ".retrieved[0].skill_id",
                )
            ]
        else:
            expected_by_id[instance_id] = []
    return expected_by_id, sha256_file(source_path)


def public_answer_provenance(
    answer: JsonObject,
    provenance_row: JsonObject,
    provenance: ProvenanceData,
    model_tag: str,
    domain: str,
    arm: ArmName,
    instance_id: str,
) -> JsonObject:
    """Validate one row label and remove private raw-source paths."""

    context: str = f"answer-provenance:{domain}:{arm}:{instance_id}"
    level: str = require_string(
        provenance_row.get("provenance_level"),
        f"{context}.provenance_level",
    )
    cohort: str = require_string(
        provenance_row.get("cohort"),
        f"{context}.cohort",
    )
    raw_structural_source: JsonValue | None = answer.get(
        "provisional_source"
    )
    structural_source: JsonObject | None = (
        None
        if raw_structural_source is None
        else require_object(
            raw_structural_source,
            f"{context}.provisional_source",
        )
    )
    early_raw_arm: bool = (
        model_tag in EARLY_RAW_MODELS
        and arm in ("routed_always", "routed_gated")
    )
    if early_raw_arm:
        if cohort != "early_raw_k2":
            raise DownstreamDataError(
                f"Early-raw cohort label mismatch: context={context}, "
                f"actual={cohort}"
            )
        if structural_source is not None and level != "posthoc_structural":
            raise DownstreamDataError(
                "Direct raw-to-schema conversion must be labeled "
                f"posthoc_structural: context={context}, level={level}"
            )
        if structural_source is None and level != (
            "formal_retry_after_import"
        ):
            raise DownstreamDataError(
                "Early-raw row without structural source must be an "
                f"explicit formal retry: context={context}, level={level}"
            )
    if level == "posthoc_structural" and structural_source is None:
        raise DownstreamDataError(
            f"posthoc_structural row lacks provisional_source: "
            f"context={context}"
        )
    if level == "formal_retry_after_import":
        if (
            model_tag != "yi15-9b"
            or not early_raw_arm
            or structural_source is not None
            or answer.get("failure_category") != "method_failure"
            or answer.get("engine_attempts") != 3
        ):
            raise DownstreamDataError(
                f"Invalid formal retry classification: context={context}"
            )
        error_payload: JsonObject = require_object(
            answer.get("error"),
            f"{context}.error",
        )
        if error_payload.get("exception_name") != "EmptyModelOutput":
            raise DownstreamDataError(
                "Formal retry must be the registered Yi EmptyModelOutput "
                f"case: context={context}, "
                f"actual={error_payload.get('exception_name')!r}"
            )
    if (
        structural_source is not None
        and level == "formal_direct"
        and model_tag != FIXED_MODEL
    ):
        raise DownstreamDataError(
            "Only Qwen reference run-history evidence may classify a row "
            "with provisional_source as formal_direct: "
            f"context={context}"
        )
    public_source: JsonObject | None = None
    if structural_source is not None:
        source_sha256: str = require_sha256(
            structural_source.get("source_sha256"),
            f"{context}.provisional_source.source_sha256",
        )
        source_line_number: int = require_integer(
            structural_source.get("source_line_number"),
            f"{context}.provisional_source.source_line_number",
        )
        source_line_sha256: str = require_sha256(
            structural_source.get("source_line_sha256"),
            f"{context}.provisional_source.source_line_sha256",
        )
        if source_sha256 not in provenance["source_line_hashes"]:
            raise DownstreamDataError(
                "Answer references a raw source not verified by the "
                f"provenance manifest: context={context}, "
                f"source_sha256={source_sha256}"
            )
        line_hashes: tuple[str, ...] = provenance["source_line_hashes"][
            source_sha256
        ]
        if (
            source_line_number < 1
            or source_line_number > len(line_hashes)
            or line_hashes[source_line_number - 1] != source_line_sha256
        ):
            raise DownstreamDataError(
                "Answer provisional source line does not match the "
                f"recovered raw file: context={context}, "
                f"line={source_line_number}"
            )
        require_string(
            structural_source.get("source_path"),
            f"{context}.provisional_source.source_path",
        )
        public_source = {
            "source_sha256": source_sha256,
            "source_line_number": source_line_number,
            "source_line_sha256": source_line_sha256,
        }
    return {
        "provenance_level": level,
        "provenance_cohort": cohort,
        "structural_source": public_source,
    }


def validate_answer_domain(
    formal_dir: Path,
    model_tag: str,
    domain: str,
    arm: ArmName,
    expected_by_id: Mapping[str, list[str]],
    retrieval: RetrievalData,
    provenance: ProvenanceData,
) -> tuple[list[JsonObject], dict[str, str]]:
    """Validate one complete answer/evaluation job and make public rows."""

    file_arm: str = arm.replace("_", "-")
    answer_path: Path = formal_dir / f"{domain}-{file_arm}.jsonl"
    evaluation_path: Path = formal_dir / f"{domain}-{file_arm}.eval.json"
    answer_rows: list[JsonObject] = load_jsonl(
        answer_path,
        f"answers:{domain}:{arm}",
    )
    answer_index: dict[str, JsonObject] = index_rows(
        answer_rows,
        f"answers:{domain}:{arm}",
    )
    verify_coverage(
        retrieval["ids"],
        list(answer_index),
        f"answers:{domain}:{arm}",
    )
    evaluation: JsonObject = load_json(
        evaluation_path,
        f"evaluation:{domain}:{arm}",
    )
    expected_evaluation_identity: dict[str, JsonValue] = {
        "schema_version": "k2-answer-evaluation-v1",
        "model": model_tag,
        "domain": domain,
        "arm": arm,
    }
    for field, expected_value in expected_evaluation_identity.items():
        if evaluation.get(field) != expected_value:
            raise DownstreamDataError(
                f"Evaluation identity mismatch: domain={domain}, arm={arm}, "
                f"field={field}, expected={expected_value!r}, "
                f"actual={evaluation.get(field)!r}"
            )
    served_model: str = require_string(
        evaluation.get("served_model"),
        f"evaluation:{domain}:{arm}.served_model",
    )
    evaluation_provenance: JsonObject = require_object(
        evaluation.get("provenance"),
        f"evaluation:{domain}:{arm}.provenance",
    )
    actual_answers_sha256: str = sha256_file(answer_path)
    if require_sha256(
        evaluation_provenance.get("answers_sha256"),
        f"evaluation:{domain}:{arm}.provenance.answers_sha256",
    ) != actual_answers_sha256:
        raise DownstreamDataError(
            f"Evaluation is not bound to the answer file: "
            f"domain={domain}, arm={arm}"
        )
    raw_details: list[JsonValue] = require_list(
        evaluation.get("details"),
        f"evaluation:{domain}:{arm}.details",
    )
    detail_rows: list[JsonObject] = [
        require_object(
            raw_detail,
            f"evaluation:{domain}:{arm}.details[{index}]",
        )
        for index, raw_detail in enumerate(raw_details)
    ]
    detail_index: dict[str, JsonObject] = index_rows(
        detail_rows,
        f"evaluation:{domain}:{arm}",
    )
    verify_coverage(
        retrieval["ids"],
        list(detail_index),
        f"evaluation:{domain}:{arm}",
    )
    public_rows: list[JsonObject] = []
    for instance_id in retrieval["ids"]:
        answer: JsonObject = answer_index[instance_id]
        detail: JsonObject = detail_index[instance_id]
        context: str = f"answer-evaluation:{domain}:{arm}:{instance_id}"
        if answer.get("schema_version") != "k2-answer-record-v1":
            raise DownstreamDataError(
                f"Answer schema mismatch: context={context}"
            )
        expected_answer_identity: dict[str, JsonValue] = {
            "instance_id": instance_id,
            "dataset": domain,
            "method": arm,
            "served_model": served_model,
            "expected_skill_ids": expected_by_id[instance_id],
        }
        for field, expected_value in expected_answer_identity.items():
            if answer.get(field) != expected_value:
                raise DownstreamDataError(
                    f"Answer identity mismatch: context={context}, "
                    f"field={field}, expected={expected_value!r}, "
                    f"actual={answer.get(field)!r}"
                )
        expected_detail_identity: dict[str, JsonValue] = {
            "schema_version": "k2-answer-evaluation-row-v1",
            "instance_id": instance_id,
            "model": model_tag,
            "served_model": served_model,
            "domain": domain,
            "arm": arm,
            "expected_skill_ids": expected_by_id[instance_id],
            "is_validation": (
                instance_id in retrieval["validation_ids"]
            ),
        }
        for field, expected_value in expected_detail_identity.items():
            if detail.get(field) != expected_value:
                raise DownstreamDataError(
                    f"Evaluation detail mismatch: context={context}, "
                    f"field={field}, expected={expected_value!r}, "
                    f"actual={detail.get(field)!r}"
                )
        category: str = require_string(
            detail.get("failure_category"),
            f"{context}.failure_category",
        )
        if category not in {"success", "method_failure"}:
            raise DownstreamDataError(
                f"Answer evaluation contains unresolved failure: "
                f"context={context}, failure_category={category}"
            )
        if answer.get("failure_category") != category:
            raise DownstreamDataError(
                f"Answer/evaluation failure category mismatch: "
                f"context={context}"
            )
        request_hash: str = require_sha256(
            detail.get("request_hash"),
            f"{context}.request_hash",
        )
        raw_answer_hash: JsonValue | None = answer.get("request_hash")
        if raw_answer_hash is not None and raw_answer_hash != request_hash:
            raise DownstreamDataError(
                f"Answer/evaluation request hash mismatch: context={context}"
            )
        skill_ids_used: list[str] = require_string_list(
            detail.get("skill_ids_used"),
            f"{context}.skill_ids_used",
        )
        if answer.get("skill_ids_used") != skill_ids_used:
            raise DownstreamDataError(
                f"Answer/evaluation loaded skills mismatch: context={context}"
            )
        if category == "success" and skill_ids_used != expected_by_id[instance_id]:
            raise DownstreamDataError(
                f"Successful answer used the wrong skills: context={context}"
            )
        provenance_key: tuple[str, str, str] = (
            domain,
            arm,
            instance_id,
        )
        if provenance_key not in provenance["rows"]:
            raise DownstreamDataError(
                f"Answer provenance row is missing: key={provenance_key}"
            )
        public_provenance: JsonObject = public_answer_provenance(
            answer,
            provenance["rows"][provenance_key],
            provenance,
            model_tag,
            domain,
            arm,
            instance_id,
        )
        public_rows.append(
            {
                "schema_version": "k2-public-answer-row-v1",
                "model": model_tag,
                "served_model": served_model,
                "domain": domain,
                "arm": arm,
                "instance_id": instance_id,
                "correct": require_boolean(
                    detail.get("correct"),
                    f"{context}.correct",
                ),
                "failure_category": category,
                "request_hash": request_hash,
                "expected_skill_ids": expected_by_id[instance_id],
                "skill_ids_used": skill_ids_used,
                "is_validation": (
                    instance_id in retrieval["validation_ids"]
                ),
                "raw_output_sha256": require_sha256(
                    detail.get("raw_output_sha256"),
                    f"{context}.raw_output_sha256",
                ),
                **public_provenance,
            }
        )
    validate_evaluation_metrics(
        evaluation,
        public_rows,
        f"evaluation:{domain}:{arm}",
    )
    return public_rows, {
        answer_path.name: actual_answers_sha256,
        evaluation_path.name: sha256_file(evaluation_path),
    }


def validate_evaluation_metrics(
    evaluation: JsonObject,
    rows: Sequence[JsonObject],
    context: str,
) -> None:
    """Recompute and verify full and held-out evaluation summaries."""

    metrics: JsonObject = require_object(
        evaluation.get("metrics"),
        f"{context}.metrics",
    )
    for split in ("full", "heldout"):
        selected: list[JsonObject] = (
            list(rows)
            if split == "full"
            else [
                row
                for row in rows
                if row.get("is_validation") is False
            ]
        )
        summary: JsonObject = require_object(
            metrics.get(split),
            f"{context}.metrics.{split}",
        )
        correct: int = sum(row.get("correct") is True for row in selected)
        if summary.get("total") != len(selected) or summary.get("correct") != correct:
            raise DownstreamDataError(
                f"Evaluation metric counts mismatch: context={context}, "
                f"split={split}"
            )
        accuracy: float = require_number(
            summary.get("accuracy"),
            f"{context}.metrics.{split}.accuracy",
        )
        expected_accuracy: float = correct / len(selected)
        if not math.isclose(accuracy, expected_accuracy, abs_tol=1e-12):
            raise DownstreamDataError(
                f"Evaluation accuracy mismatch: context={context}, "
                f"split={split}, expected={expected_accuracy}, "
                f"actual={accuracy}"
            )


def validate_reuse_job(
    formal_dir: Path,
    model_tag: str,
    domain: str,
    arm: ArmName,
    expected_by_id: Mapping[str, list[str]],
    answer_source_sha256: str,
) -> JsonObject:
    """Validate answer reuse/completion audits and return a public summary."""

    file_arm: str = arm.replace("_", "-")
    reuse_path: Path = (
        formal_dir / "audits" / f"{domain}-{file_arm}.reuse.jsonl"
    )
    completion_path: Path = (
        formal_dir / "audits" / f"{domain}-{file_arm}.completion.json"
    )
    reuse_rows: list[JsonObject] = load_jsonl(
        reuse_path,
        f"reuse:{domain}:{arm}",
    )
    reuse_index: dict[str, JsonObject] = index_rows(
        reuse_rows,
        f"reuse:{domain}:{arm}",
    )
    verify_coverage(
        tuple(expected_by_id),
        list(reuse_index),
        f"reuse:{domain}:{arm}",
    )
    status_counts: dict[str, int] = {
        "reused_same_arm": 0,
        "needs_inference": 0,
        "rejected": 0,
    }
    reason_counts: dict[str, int] = {}
    for instance_id, expected_skill_ids in expected_by_id.items():
        row: JsonObject = reuse_index[instance_id]
        if row.get("arm") != arm:
            raise DownstreamDataError(
                f"Reuse arm mismatch: domain={domain}, arm={arm}, "
                f"instance_id={instance_id}, actual={row.get('arm')!r}"
            )
        if row.get("expected_skill_ids") != expected_skill_ids:
            raise DownstreamDataError(
                f"Reuse expected skills mismatch: domain={domain}, arm={arm}, "
                f"instance_id={instance_id}"
            )
        status: str = require_string(
            row.get("status"),
            f"reuse:{domain}:{arm}:{instance_id}.status",
        )
        if status not in status_counts:
            raise DownstreamDataError(
                f"Unknown reuse status: domain={domain}, arm={arm}, "
                f"instance_id={instance_id}, status={status}"
            )
        require_sha256(
            row.get("new_request_hash"),
            f"reuse:{domain}:{arm}:{instance_id}.new_request_hash",
        )
        reason: str = require_string(
            row.get("reason"),
            f"reuse:{domain}:{arm}:{instance_id}.reason",
        )
        status_counts[status] += 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    completion: JsonObject = load_json(
        completion_path,
        f"completion:{domain}:{arm}",
    )
    expected_completion: dict[str, JsonValue] = {
        "schema_version": "k2-answer-validation-v1",
        "valid": True,
        "model": CACHE_MODEL_TAGS[model_tag],
        "domain": domain,
        "arm": arm,
        "expected": len(expected_by_id),
        "observed": len(expected_by_id),
        "reused_same_arm": status_counts["reused_same_arm"],
        "new_records": status_counts["needs_inference"]
        + status_counts["rejected"],
        "answers_sha256": answer_source_sha256,
        "audit_sha256": sha256_file(reuse_path),
    }
    for field, expected_value in expected_completion.items():
        if completion.get(field) != expected_value:
            raise DownstreamDataError(
                f"Completion audit mismatch: domain={domain}, arm={arm}, "
                f"field={field}, expected={expected_value!r}, "
                f"actual={completion.get(field)!r}"
            )
    return {
        "schema_version": "k2-public-reuse-job-v1",
        "domain": domain,
        "arm": arm,
        "instances": len(expected_by_id),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "reuse_audit_sha256": sha256_file(reuse_path),
        "completion_sha256": sha256_file(completion_path),
        "answer_source_sha256": answer_source_sha256,
    }


def load_formal_data(
    formal_dir: Path,
    model_tag: str,
    retrievals: Mapping[str, RetrievalData],
    gates: Mapping[str, GateData],
    fixed_sources: Mapping[str, Path],
    provenance: ProvenanceData,
) -> FormalData:
    """Validate formal loading, selection, answers, and reuse evidence."""

    marker_evidence: JsonObject = validate_formal_root(
        formal_dir,
        model_tag,
    )
    expected_provenance_keys: set[tuple[str, str, str]] = {
        (domain, arm, instance_id)
        for domain in RULE_DOMAINS
        for arm in expected_answer_arms(model_tag)
        for instance_id in retrievals[domain]["ids"]
    }
    actual_provenance_keys: set[tuple[str, str, str]] = set(
        provenance["rows"]
    )
    if actual_provenance_keys != expected_provenance_keys:
        raise DownstreamDataError(
            "Answer provenance support mismatch: "
            f"missing={sorted(expected_provenance_keys - actual_provenance_keys)[:20]}, "
            f"unexpected={sorted(actual_provenance_keys - expected_provenance_keys)[:20]}"
        )
    loading_rows: list[JsonObject] = []
    selection_rows: list[JsonObject] = []
    answer_rows: list[JsonObject] = []
    reuse_jobs: list[JsonObject] = []
    formal_source_hashes: dict[str, str] = {}
    for domain in RULE_DOMAINS:
        retrieval: RetrievalData = retrievals[domain]
        gate: GateData = gates[domain]
        selected_by_id: dict[str, list[str]] = {}
        selected_source_sha256: str = ""
        if model_tag in SELECT_ELIGIBLE_MODELS:
            (
                domain_selection_rows,
                selected_by_id,
                selected_source_sha256,
                selection_completion_sha256,
            ) = validate_selection_domain(
                formal_dir,
                model_tag,
                domain,
                retrieval,
            )
            selection_rows.extend(domain_selection_rows)
            formal_source_hashes[
                f"{domain}-routed-select.selection.jsonl"
            ] = sha256_file(
                formal_dir / f"{domain}-routed-select.selection.jsonl"
            )
            formal_source_hashes[
                f"{domain}-routed-select-source.json"
            ] = selected_source_sha256
            formal_source_hashes[
                "audits/"
                f"{domain}-routed-select.selection-completion.json"
            ] = selection_completion_sha256

        deterministic_path: Path = (
            formal_dir / f"{domain}-routed-always-gated.loading.jsonl"
        )
        deterministic_rows: list[JsonObject] = load_jsonl(
            deterministic_path,
            f"loading:{domain}:deterministic",
        )
        deterministic_index: dict[
            tuple[str, str], JsonObject
        ] = {}
        for row in deterministic_rows:
            instance_id: str = require_string(
                row.get("instance_id"),
                f"loading:{domain}:instance_id",
            )
            arm: str = require_string(
                row.get("arm"),
                f"loading:{domain}:{instance_id}.arm",
            )
            key: tuple[str, str] = (arm, instance_id)
            if key in deterministic_index:
                raise DownstreamDataError(
                    f"Duplicate loading row: domain={domain}, key={key}"
                )
            deterministic_index[key] = row
        expected_deterministic_keys: set[tuple[str, str]] = {
            (arm, instance_id)
            for arm in ("routed_always", "routed_gated")
            for instance_id in retrieval["ids"]
        }
        if set(deterministic_index) != expected_deterministic_keys:
            raise DownstreamDataError(
                f"Deterministic loading support mismatch: domain={domain}"
            )
        for arm, source_sha256 in (
            ("routed_always", retrieval["source_sha256"]),
            ("routed_gated", gate["source_sha256"]),
        ):
            for instance_id in retrieval["ids"]:
                loading_rows.append(
                    validate_loading_row(
                        deterministic_index[(arm, instance_id)],
                        model_tag,
                        domain,
                        arm,
                        expected_routed_skill_ids(
                            arm,
                            instance_id,
                            retrieval,
                            gate,
                            selected_by_id,
                        ),
                        retrieval["gold_by_id"][instance_id],
                        instance_id in retrieval["validation_ids"],
                        source_sha256,
                        f"loading:{domain}:{arm}:{instance_id}",
                    )
                )
        formal_source_hashes[
            deterministic_path.name
        ] = sha256_file(deterministic_path)

        if model_tag in SELECT_ELIGIBLE_MODELS:
            select_loading_path: Path = (
                formal_dir / f"{domain}-routed-select.loading.jsonl"
            )
            select_loading_input: list[JsonObject] = load_jsonl(
                select_loading_path,
                f"loading:{domain}:routed_select",
            )
            select_loading_index: dict[str, JsonObject] = index_rows(
                select_loading_input,
                f"loading:{domain}:routed_select",
            )
            verify_coverage(
                retrieval["ids"],
                list(select_loading_index),
                f"loading:{domain}:routed_select",
            )
            for instance_id in retrieval["ids"]:
                loading_rows.append(
                    validate_loading_row(
                        select_loading_index[instance_id],
                        model_tag,
                        domain,
                        "routed_select",
                        selected_by_id[instance_id],
                        retrieval["gold_by_id"][instance_id],
                        instance_id in retrieval["validation_ids"],
                        selected_source_sha256,
                        f"loading:{domain}:routed_select:{instance_id}",
                    )
                )
            formal_source_hashes[
                select_loading_path.name
            ] = sha256_file(select_loading_path)

        fixed_by_id: dict[str, list[str]] = {}
        if model_tag == FIXED_MODEL:
            fixed_by_id, _ = validate_fixed_source(
                fixed_sources[domain],
                model_tag,
                domain,
                retrieval,
            )

        for arm in expected_answer_arms(model_tag):
            if arm == "fixed_gated":
                expected_by_id: Mapping[str, list[str]] = fixed_by_id
            else:
                expected_by_id = {
                    instance_id: expected_routed_skill_ids(
                        arm,
                        instance_id,
                        retrieval,
                        gate,
                        selected_by_id,
                    )
                    for instance_id in retrieval["ids"]
                }
            domain_answer_rows, source_hashes = validate_answer_domain(
                formal_dir,
                model_tag,
                domain,
                arm,
                expected_by_id,
                retrieval,
                provenance,
            )
            answer_rows.extend(domain_answer_rows)
            formal_source_hashes.update(source_hashes)
            answer_filename: str = (
                f"{domain}-{arm.replace('_', '-')}.jsonl"
            )
            reuse_jobs.append(
                validate_reuse_job(
                    formal_dir,
                    model_tag,
                    domain,
                    arm,
                    expected_by_id,
                    formal_source_hashes[answer_filename],
                )
            )
    marker_path: Path = formal_dir / "FORMAL_COMPLETE"
    if marker_path.is_file():
        formal_source_hashes["FORMAL_COMPLETE"] = sha256_file(marker_path)
    selection_completion_audits: int = (
        len(RULE_DOMAINS)
        if model_tag in SELECT_ELIGIBLE_MODELS
        else 0
    )
    formal_completion_evidence: JsonObject = {
        "schema_version": "k2-public-formal-completion-evidence-v1",
        "policy": (
            "producer_marker_plus_per_job_completion_audits"
            if marker_evidence["present"]
            else "per_job_completion_audits"
        ),
        "marker": marker_evidence,
        "validated_answer_completion_audits": len(reuse_jobs),
        "validated_selection_completion_audits": (
            selection_completion_audits
        ),
    }
    reuse_manifest: JsonObject = {
        "schema_version": "k2-public-reuse-manifest-v1",
        "model": model_tag,
        "formal_completion_evidence": formal_completion_evidence,
        "jobs": sorted(
            reuse_jobs,
            key=lambda row: (
                RULE_DOMAINS.index(
                    require_string(row.get("domain"), "reuse.domain")
                ),
                ANSWER_ARM_ORDER.index(
                    require_string(row.get("arm"), "reuse.arm")
                ),
            ),
        ),
    }
    return {
        "loading_rows": loading_rows,
        "selection_rows": selection_rows,
        "answer_rows": answer_rows,
        "reuse_manifest": reuse_manifest,
        "formal_source_hashes": formal_source_hashes,
        "formal_completion_evidence": formal_completion_evidence,
        "provenance_summary": provenance["summary"],
    }


def split_rows(
    rows: Sequence[JsonObject],
    split: str,
) -> list[JsonObject]:
    """Return full or held-out rows."""

    if split == "full":
        return list(rows)
    if split == "heldout":
        return [
            row for row in rows if row.get("is_validation") is False
        ]
    raise DownstreamDataError(f"Unknown metric split: split={split}")


def expected_loading_metric_keys(
    model_tag: str,
) -> set[tuple[str, str, str | None, str]]:
    """Return required per-model loading metric keys."""

    arms: tuple[str, ...] = (
        ROUTED_ARMS
        if model_tag in SELECT_ELIGIBLE_MODELS
        else ROUTED_ARMS[:2]
    )
    return {
        (level, split, domain, arm)
        for split in ("full", "heldout")
        for arm in arms
        for level, domain in (
            ("per_model_pooled", None),
            *(
                ("per_model_domain", rule_domain)
                for rule_domain in RULE_DOMAINS
            ),
        )
    }


def validate_loading_metric_row(
    row: JsonObject,
    decision_rows: Sequence[JsonObject],
    model_tag: str,
    context: str,
) -> None:
    """Recompute one loading metric record from public decisions."""

    if row.get("schema_version") != "k2-loading-metrics-v1":
        raise DownstreamDataError(
            f"Loading metric schema mismatch: context={context}"
        )
    level: str = require_string(row.get("level"), f"{context}.level")
    split: str = require_string(row.get("split"), f"{context}.split")
    arm: str = require_string(row.get("arm"), f"{context}.arm")
    domain_value: JsonValue | None = row.get("domain")
    domain: str | None = (
        None
        if domain_value is None
        else require_string(domain_value, f"{context}.domain")
    )
    selected: list[JsonObject] = [
        decision
        for decision in split_rows(decision_rows, split)
        if decision.get("model") == model_tag
        and decision.get("arm") == arm
        and (domain is None or decision.get("domain") == domain)
    ]
    typed_rows = cast(Sequence, selected)
    expected = compute_loading_metrics(typed_rows)
    for field, expected_value in expected.items():
        actual_value: JsonValue | None = row.get(field)
        if isinstance(expected_value, float):
            if not isinstance(actual_value, (int, float)) or not math.isclose(
                float(actual_value),
                expected_value,
                abs_tol=1e-12,
            ):
                raise DownstreamDataError(
                    f"Loading metric mismatch: context={context}, "
                    f"field={field}, expected={expected_value}, "
                    f"actual={actual_value!r}"
                )
        elif actual_value != expected_value:
            raise DownstreamDataError(
                f"Loading metric mismatch: context={context}, field={field}, "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    if level not in {"per_model_pooled", "per_model_domain"}:
        raise DownstreamDataError(
            f"Unexpected per-model loading metric level: context={context}, "
            f"level={level}"
        )


def expected_answer_metric_keys(
    model_tag: str,
) -> set[tuple[str, str, str | None, str]]:
    """Return required per-model answer metric keys."""

    return {
        (level, split, domain, arm)
        for split in ("full", "heldout")
        for arm in expected_answer_arms(model_tag)
        for level, domain in (
            ("model_pooled", None),
            *(
                ("model_domain", rule_domain)
                for rule_domain in RULE_DOMAINS
            ),
        )
    }


def validate_answer_metric_row(
    row: JsonObject,
    answer_rows: Sequence[JsonObject],
    model_tag: str,
    context: str,
) -> None:
    """Recompute one answer metric record from public answer rows."""

    if row.get("schema_version") != "k2-answer-metrics-long-v1":
        raise DownstreamDataError(
            f"Answer metric schema mismatch: context={context}"
        )
    level: str = require_string(row.get("level"), f"{context}.level")
    split: str = require_string(row.get("split"), f"{context}.split")
    arm: str = require_string(row.get("arm"), f"{context}.arm")
    raw_domain: JsonValue | None = row.get("domain")
    domain: str | None = (
        None
        if raw_domain is None
        else require_string(raw_domain, f"{context}.domain")
    )
    selected: list[JsonObject] = [
        answer
        for answer in split_rows(answer_rows, split)
        if answer.get("model") == model_tag
        and answer.get("arm") == arm
        and (domain is None or answer.get("domain") == domain)
    ]
    correct: int = sum(answer.get("correct") is True for answer in selected)
    expected_values: dict[str, JsonValue] = {
        "n": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected),
    }
    for field, expected_value in expected_values.items():
        actual_value: JsonValue | None = row.get(field)
        if isinstance(expected_value, float):
            if not isinstance(actual_value, (int, float)) or not math.isclose(
                float(actual_value),
                expected_value,
                abs_tol=1e-12,
            ):
                raise DownstreamDataError(
                    f"Answer metric mismatch: context={context}, "
                    f"field={field}, expected={expected_value}, "
                    f"actual={actual_value!r}"
                )
        elif actual_value != expected_value:
            raise DownstreamDataError(
                f"Answer metric mismatch: context={context}, field={field}, "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    if level not in {"model_pooled", "model_domain"}:
        raise DownstreamDataError(
            f"Unexpected per-model answer metric level: context={context}, "
            f"level={level}"
        )


def metric_key(
    row: JsonObject,
    context: str,
) -> tuple[str, str, str | None, str]:
    """Return the unique long-metric key."""

    raw_domain: JsonValue | None = row.get("domain")
    domain: str | None = (
        None
        if raw_domain is None
        else require_string(raw_domain, f"{context}.domain")
    )
    return (
        require_string(row.get("level"), f"{context}.level"),
        require_string(row.get("split"), f"{context}.split"),
        domain,
        require_string(row.get("arm"), f"{context}.arm"),
    )


def load_metric_data(
    loading_metrics_path: Path,
    answer_metrics_path: Path,
    answer_summary_path: Path,
    model_tag: str,
    formal: FormalData,
) -> MetricData:
    """Validate and subset fleet long metrics for one model pack."""

    all_loading_rows: list[JsonObject] = load_jsonl(
        loading_metrics_path,
        "fleet-loading-metrics",
    )
    loading_rows: list[JsonObject] = [
        row for row in all_loading_rows if row.get("model") == model_tag
    ]
    loading_index: dict[
        tuple[str, str, str | None, str], JsonObject
    ] = {}
    for row_index, row in enumerate(loading_rows):
        key = metric_key(row, f"loading-metric[{row_index}]")
        if key in loading_index:
            raise DownstreamDataError(
                f"Duplicate loading metric row: model={model_tag}, key={key}"
            )
        loading_index[key] = row
        validate_loading_metric_row(
            row,
            formal["loading_rows"],
            model_tag,
            f"loading-metric:{key}",
        )
    if set(loading_index) != expected_loading_metric_keys(model_tag):
        raise DownstreamDataError(
            "Per-model loading metric coverage mismatch: "
            f"model={model_tag}, "
            f"missing={sorted(expected_loading_metric_keys(model_tag) - set(loading_index), key=str)}, "
            f"unexpected={sorted(set(loading_index) - expected_loading_metric_keys(model_tag), key=str)}"
        )

    all_answer_rows: list[JsonObject] = load_jsonl(
        answer_metrics_path,
        "fleet-answer-metrics",
    )
    answer_rows: list[JsonObject] = [
        row for row in all_answer_rows if row.get("model") == model_tag
    ]
    answer_index: dict[
        tuple[str, str, str | None, str], JsonObject
    ] = {}
    for row_index, row in enumerate(answer_rows):
        key = metric_key(row, f"answer-metric[{row_index}]")
        if key in answer_index:
            raise DownstreamDataError(
                f"Duplicate answer metric row: model={model_tag}, key={key}"
            )
        answer_index[key] = row
        validate_answer_metric_row(
            row,
            formal["answer_rows"],
            model_tag,
            f"answer-metric:{key}",
        )
    if set(answer_index) != expected_answer_metric_keys(model_tag):
        raise DownstreamDataError(
            "Per-model answer metric coverage mismatch: "
            f"model={model_tag}, "
            f"missing={sorted(expected_answer_metric_keys(model_tag) - set(answer_index), key=str)}, "
            f"unexpected={sorted(set(answer_index) - expected_answer_metric_keys(model_tag), key=str)}"
        )

    summary: JsonObject = load_json(
        answer_summary_path,
        "fleet-answer-summary",
    )
    if summary.get("schema_version") != "k2-answer-summary-v1":
        raise DownstreamDataError(
            "Answer summary schema mismatch: "
            f"actual={summary.get('schema_version')!r}"
        )
    model_pooled: JsonObject = require_object(
        summary.get("model_pooled"),
        "fleet-answer-summary.model_pooled",
    )
    model_block: JsonObject = require_object(
        model_pooled.get(model_tag),
        f"fleet-answer-summary.model_pooled.{model_tag}",
    )
    for split in ("full", "heldout"):
        split_block: JsonObject = require_object(
            model_block.get(split),
            f"fleet-answer-summary.model_pooled.{model_tag}.{split}",
        )
        for arm in expected_answer_arms(model_tag):
            summary_row: JsonObject = require_object(
                split_block.get(arm),
                f"fleet-answer-summary.model_pooled."
                f"{model_tag}.{split}.{arm}",
            )
            long_row: JsonObject = answer_index[
                ("model_pooled", split, None, arm)
            ]
            if canonical_json(summary_row) != canonical_json(long_row):
                raise DownstreamDataError(
                    "Answer summary disagrees with long metrics: "
                    f"model={model_tag}, split={split}, arm={arm}"
                )
    answer_metrics: JsonObject = {
        "schema_version": "k2-public-answer-metrics-v1",
        "model": model_tag,
        "model_pooled": model_block,
        "answer_metrics_long_sha256": sha256_file(answer_metrics_path),
        "answer_summary_sha256": sha256_file(answer_summary_path),
    }
    flat_rows: list[JsonObject] = [
        {**row, "metric_family": "loading"}
        for row in loading_rows
    ] + [
        {**row, "metric_family": "answer"}
        for row in answer_rows
    ]
    return {
        "answer_metrics": answer_metrics,
        "flat_rows": flat_rows,
    }


def expected_significance_names(model_tag: str) -> tuple[str, ...]:
    """Return the registered comparisons involving one model."""

    names: list[str] = ["gated_vs_always_seven_model"]
    if model_tag in SELECT_ELIGIBLE_MODELS:
        names.extend(
            [
                "gated_vs_select_five_model",
                "select_vs_always_five_model",
            ]
        )
    if model_tag == FIXED_MODEL:
        names.append("routed_gated_vs_fixed_gated")
    return tuple(names)


def load_significance(path: Path, model_tag: str) -> JsonObject:
    """Validate and subset registered paired comparisons for one model."""

    payload: JsonObject = load_json(path, "fleet-significance")
    if payload.get("schema_version") != "k2-answer-paired-comparisons-v1":
        raise DownstreamDataError(
            "Significance schema mismatch: "
            f"actual={payload.get('schema_version')!r}"
        )
    comparisons: JsonObject = require_object(
        payload.get("comparisons"),
        "fleet-significance.comparisons",
    )
    selected: JsonObject = {}
    for name in expected_significance_names(model_tag):
        comparison: JsonObject = require_object(
            comparisons.get(name),
            f"fleet-significance.comparisons.{name}",
        )
        by_model: JsonObject = require_object(
            comparison.get("by_model"),
            f"fleet-significance.comparisons.{name}.by_model",
        )
        model_statistics: JsonObject = require_object(
            by_model.get(model_tag),
            f"fleet-significance.comparisons.{name}"
            f".by_model.{model_tag}",
        )
        selected[name] = {
            "schema_version": require_string(
                comparison.get("schema_version"),
                f"fleet-significance.comparisons.{name}.schema_version",
            ),
            "split": require_string(
                comparison.get("split"),
                f"fleet-significance.comparisons.{name}.split",
            ),
            "arm_a": require_string(
                comparison.get("arm_a"),
                f"fleet-significance.comparisons.{name}.arm_a",
            ),
            "arm_b": require_string(
                comparison.get("arm_b"),
                f"fleet-significance.comparisons.{name}.arm_b",
            ),
            "statistics": model_statistics,
        }
    return {
        "schema_version": "k2-public-significance-v1",
        "model": model_tag,
        "split": require_string(
            payload.get("split"),
            "fleet-significance.split",
        ),
        "bootstrap_samples": require_integer(
            payload.get("bootstrap_samples"),
            "fleet-significance.bootstrap_samples",
        ),
        "bootstrap_seed": require_integer(
            payload.get("bootstrap_seed"),
            "fleet-significance.bootstrap_seed",
        ),
        "comparisons": selected,
        "source_sha256": sha256_file(path),
    }


def domain_sort_key(row: JsonObject) -> int:
    """Return the frozen domain order for one row."""

    domain: str = require_string(row.get("domain"), "sort.domain")
    return RETRIEVAL_DOMAINS.index(domain)


def instance_sort_key(row: JsonObject) -> tuple[int, int, str]:
    """Return domain, arm, and instance ordering."""

    raw_arm: JsonValue | None = row.get("arm")
    arm_index: int = (
        -1
        if raw_arm is None
        else ANSWER_ARM_ORDER.index(require_string(raw_arm, "sort.arm"))
    )
    return (
        domain_sort_key(row),
        arm_index,
        require_string(row.get("instance_id"), "sort.instance_id"),
    )


def metric_sort_key(
    row: JsonObject,
) -> tuple[str, str, str, str, str]:
    """Return deterministic ordering for mixed metric families."""

    raw_domain: JsonValue | None = row.get("domain")
    domain: str = "" if raw_domain is None else require_string(
        raw_domain,
        "metric-sort.domain",
    )
    return (
        require_string(row.get("metric_family"), "metric-sort.family"),
        require_string(row.get("split"), "metric-sort.split"),
        require_string(row.get("level"), "metric-sort.level"),
        require_string(row.get("arm"), "metric-sort.arm"),
        domain,
    )


def write_json_file(path: Path, payload: JsonLike) -> None:
    """Write one canonical JSON value."""

    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def write_gzip_jsonl(path: Path, rows: Sequence[JsonObject]) -> None:
    """Write reproducible canonical JSONL gzip bytes."""

    with path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            mtime=0,
        ) as gzip_file:
            for row in rows:
                gzip_file.write((canonical_json(row) + "\n").encode("utf-8"))


def arm_availability(model_tag: str) -> JsonObject:
    """Describe available, unavailable, and non-applicable arms."""

    if model_tag in SELECT_ELIGIBLE_MODELS:
        select: JsonObject = {
            "status": "available",
            "domains": list(RULE_DOMAINS),
        }
    else:
        select = {
            "status": "unavailable",
            "domains": [],
            "reason": SELECT_INELIGIBLE_REASONS[model_tag],
            "accounting": "excluded from Select denominators; never zero-filled",
        }
    fixed: JsonObject = (
        {
            "status": "available",
            "domains": list(RULE_DOMAINS),
        }
        if model_tag == FIXED_MODEL
        else {
            "status": "not_applicable",
            "domains": [],
            "reason": "The fixed naive_skill component arm is Qwen4-only.",
        }
    )
    return {
        "routed_always": {
            "status": "available",
            "domains": list(RULE_DOMAINS),
        },
        "routed_gated": {
            "status": "available",
            "domains": list(RULE_DOMAINS),
        },
        "routed_select": select,
        "fixed_gated": fixed,
    }


def readme_text(
    model_tag: str,
    retrieval_rows: int,
    loading_rows: int,
    answer_rows: int,
) -> str:
    """Build a concise deterministic public-pack README."""

    select_line: str = (
        "- `routed_select`: available on the four rule-scored domains."
        if model_tag in SELECT_ELIGIBLE_MODELS
        else (
            "- `routed_select`: unavailable. "
            + SELECT_INELIGIBLE_REASONS[model_tag]
        )
    )
    fixed_line: str = (
        "- `fixed_gated`: available as the Qwen4 component ablation."
        if model_tag == FIXED_MODEL
        else "- `fixed_gated`: not applicable to this model."
    )
    return "\n".join(
        [
            f"# K=2 public result pack: {model_tag}",
            "",
            "This directory is the active unified K=2 evidence pack.",
            "All JSONL gzip files use canonical JSON and reproducible gzip metadata.",
            "",
            "## Coverage",
            "",
            f"- Retrieval rows: {retrieval_rows} across five domains.",
            f"- Loading rows: {loading_rows} across routed loading arms.",
            f"- Answer rows: {answer_rows} across active answer arms.",
            "- `routed_always`: available on the four rule-scored domains.",
            "- `routed_gated`: available on the four rule-scored domains.",
            select_line,
            fixed_line,
            "",
            "## Evidence policy",
            "",
            "Per-instance outputs omit raw model text, private endpoints, tokens, "
            "checkpoints, and server paths.",
            "The manifest records hashes, row counts, schemas, provenance levels, "
            "and the explicit self-hash exclusion rule.",
            "Formal completion is verified from every per-job completion audit; "
            "the manifest also records whether an aggregate producer marker was "
            "available.",
            "",
        ]
    )


def output_record(
    directory: Path,
    filename: str,
    rows: int,
    schema: str,
    provenance_level: str,
) -> JsonObject:
    """Build one manifest record from a generated file."""

    path: Path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Generated public-pack file is missing: path={path}"
        )
    return {
        "sha256": sha256_file(path),
        "rows": rows,
        "schema": schema,
        "provenance_level": provenance_level,
    }


def source_manifest(
    args: argparse.Namespace,
    model_tag: str,
    formal: FormalData,
) -> JsonObject:
    """Record hashes of explicit inputs without publishing server paths."""

    retrieval_sources: JsonObject = {}
    router_sources: JsonObject = {}
    gate_sources: JsonObject = {}
    for domain in RETRIEVAL_DOMAINS:
        retrieval_path: Path = argument_path(args, domain, "retrieval")
        router_path: Path = argument_path(args, domain, "router")
        retrieval_sources[domain] = {
            "filename": retrieval_path.name,
            "sha256": sha256_file(retrieval_path),
        }
        router_sources[domain] = {
            "filename": router_path.name,
            "sha256": sha256_file(router_path),
        }
    for domain in RULE_DOMAINS:
        domain_sources: JsonObject = {}
        for suffix in ("signals", "taus", "gated"):
            path: Path = argument_path(args, domain, suffix)
            domain_sources[suffix] = {
                "filename": path.name,
                "sha256": sha256_file(path),
            }
        gate_sources[domain] = domain_sources
    fixed_sources: JsonObject = {}
    for domain, source_path in fixed_source_paths(args, model_tag).items():
        fixed_sources[domain] = {
            "filename": source_path.name,
            "sha256": sha256_file(source_path),
        }
    fleet_sources: JsonObject = {}
    for argument_name in (
        "loading_metrics_long",
        "answer_metrics_long",
        "answer_summary",
        "significance",
        "answer_provenance",
    ):
        raw_path: object = getattr(args, argument_name)
        if not isinstance(raw_path, Path):
            raise TypeError(
                f"Fleet CLI input is not a Path: argument={argument_name}"
            )
        path = raw_path.resolve()
        fleet_sources[argument_name] = {
            "filename": path.name,
            "sha256": sha256_file(path),
        }
    return {
        "retrieval": retrieval_sources,
        "router": router_sources,
        "gate": gate_sources,
        "fixed_gated": fixed_sources,
        "formal": {
            filename: digest
            for filename, digest in sorted(
                formal["formal_source_hashes"].items()
            )
        },
        "fleet": fleet_sources,
    }


def build_pack(
    output_dir: Path,
    args: argparse.Namespace,
    model_tag: str,
    retrieval_rows: Sequence[JsonObject],
    router_payload: JsonObject,
    gating_rows: Sequence[JsonObject],
    formal: FormalData,
    metrics: MetricData,
    significance: JsonObject,
) -> None:
    """Write and atomically publish one complete public pack."""

    output_dir_parent: Path = output_dir.parent
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing output directory: "
            f"path={output_dir}"
        )
    if not output_dir_parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: path={output_dir_parent}"
        )
    ordered_retrieval: list[JsonObject] = sorted(
        retrieval_rows,
        key=instance_sort_key,
    )
    ordered_gating: list[JsonObject] = sorted(
        gating_rows,
        key=instance_sort_key,
    )
    ordered_loading: list[JsonObject] = sorted(
        formal["loading_rows"],
        key=instance_sort_key,
    )
    ordered_selection: list[JsonObject] = sorted(
        formal["selection_rows"],
        key=instance_sort_key,
    )
    ordered_answers: list[JsonObject] = sorted(
        formal["answer_rows"],
        key=instance_sort_key,
    )
    ordered_metrics: list[JsonObject] = sorted(
        metrics["flat_rows"],
        key=metric_sort_key,
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.tmp.",
        dir=output_dir_parent,
    ) as temporary_name:
        temporary_dir: Path = Path(temporary_name)
        write_gzip_jsonl(
            temporary_dir / "retrieval_top50.jsonl.gz",
            ordered_retrieval,
        )
        write_json_file(
            temporary_dir / "router_decisions.json",
            router_payload,
        )
        write_gzip_jsonl(
            temporary_dir / "gating_per_instance.jsonl.gz",
            ordered_gating,
        )
        write_gzip_jsonl(
            temporary_dir / "loading_per_instance.jsonl.gz",
            ordered_loading,
        )
        if model_tag in SELECT_ELIGIBLE_MODELS:
            write_gzip_jsonl(
                temporary_dir / "selection_per_instance.jsonl.gz",
                ordered_selection,
            )
        elif ordered_selection:
            raise DownstreamDataError(
                f"Select-ineligible model produced selection rows: "
                f"model={model_tag}, rows={len(ordered_selection)}"
            )
        write_gzip_jsonl(
            temporary_dir / "answer_per_instance.jsonl.gz",
            ordered_answers,
        )
        write_json_file(
            temporary_dir / "answer_metrics.json",
            metrics["answer_metrics"],
        )
        write_gzip_jsonl(
            temporary_dir / "metrics_flat.jsonl.gz",
            ordered_metrics,
        )
        write_json_file(
            temporary_dir / "significance.json",
            significance,
        )
        write_json_file(
            temporary_dir / "reuse_manifest.json",
            formal["reuse_manifest"],
        )
        readme: str = readme_text(
            model_tag,
            len(ordered_retrieval),
            len(ordered_loading),
            len(ordered_answers),
        )
        (temporary_dir / "README.md").write_text(
            readme,
            encoding="utf-8",
        )
        files: JsonObject = {
            "retrieval_top50.jsonl.gz": output_record(
                temporary_dir,
                "retrieval_top50.jsonl.gz",
                len(ordered_retrieval),
                "k2-public-retrieval-row-v1",
                "per_instance",
            ),
            "router_decisions.json": output_record(
                temporary_dir,
                "router_decisions.json",
                len(RETRIEVAL_DOMAINS),
                "k2-public-router-decisions-v1",
                "per_domain",
            ),
            "gating_per_instance.jsonl.gz": output_record(
                temporary_dir,
                "gating_per_instance.jsonl.gz",
                len(ordered_gating),
                "k2-public-gating-row-v1",
                "per_instance",
            ),
            "loading_per_instance.jsonl.gz": output_record(
                temporary_dir,
                "loading_per_instance.jsonl.gz",
                len(ordered_loading),
                "k2-loading-decision-v1",
                "per_instance",
            ),
            "answer_per_instance.jsonl.gz": output_record(
                temporary_dir,
                "answer_per_instance.jsonl.gz",
                len(ordered_answers),
                "k2-public-answer-row-v1",
                "per_instance",
            ),
            "answer_metrics.json": output_record(
                temporary_dir,
                "answer_metrics.json",
                1,
                "k2-public-answer-metrics-v1",
                "per_model",
            ),
            "metrics_flat.jsonl.gz": output_record(
                temporary_dir,
                "metrics_flat.jsonl.gz",
                len(ordered_metrics),
                "mixed-k2-metrics-v1",
                "per_model",
            ),
            "significance.json": output_record(
                temporary_dir,
                "significance.json",
                1,
                "k2-public-significance-v1",
                "per_model",
            ),
            "reuse_manifest.json": output_record(
                temporary_dir,
                "reuse_manifest.json",
                1,
                "k2-public-reuse-manifest-v1",
                "audit",
            ),
            "README.md": output_record(
                temporary_dir,
                "README.md",
                len(readme.splitlines()),
                "markdown-v1",
                "documentation",
            ),
        }
        if model_tag in SELECT_ELIGIBLE_MODELS:
            files["selection_per_instance.jsonl.gz"] = output_record(
                temporary_dir,
                "selection_per_instance.jsonl.gz",
                len(ordered_selection),
                "k2-public-selection-row-v1",
                "per_instance",
            )
        manifest: JsonObject = {
            "schema_version": PUBLIC_PACK_SCHEMA,
            "model": model_tag,
            "cache_model_tag": CACHE_MODEL_TAGS[model_tag],
            "k_samples": 2,
            "retrieval_domains": list(RETRIEVAL_DOMAINS),
            "answer_domains": list(RULE_DOMAINS),
            "arms": arm_availability(model_tag),
            "answer_provenance": formal["provenance_summary"],
            "formal_completion": formal[
                "formal_completion_evidence"
            ],
            "files": {
                filename: files[filename]
                for filename in sorted(files)
            },
            "sources": source_manifest(args, model_tag, formal),
            "manifest_self_policy": {
                "included_in_files": False,
                "reason": (
                    "manifest.json is excluded because a file cannot contain "
                    "its own stable SHA-256 digest."
                ),
            },
        }
        write_json_file(temporary_dir / "manifest.json", manifest)
        expected_names: set[str] = {
            *files,
            "manifest.json",
        }
        actual_names: set[str] = {
            entry.name for entry in temporary_dir.iterdir()
        }
        if actual_names != expected_names:
            raise DownstreamDataError(
                "Generated public-pack file set mismatch: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        os.rename(temporary_dir, output_dir)


def main() -> None:
    """Validate all evidence and export one immutable K=2 pack."""

    args = parse_args()
    model_tag: str = str(args.model_tag)
    if model_tag not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported K=2 result tag: model={model_tag}, "
            f"supported={sorted(SUPPORTED_MODELS)}"
        )
    formal_dir: Path = cast(Path, args.formal_dir).resolve()
    output_dir: Path = cast(Path, args.output_dir).resolve()
    fixed_sources: dict[str, Path] = fixed_source_paths(args, model_tag)
    retrievals, router_payload, retrieval_rows = load_retrieval_sources(
        args,
        model_tag,
    )
    gates, gating_rows = load_gate_sources(
        args,
        model_tag,
        retrievals,
    )
    provenance: ProvenanceData = load_answer_provenance(
        cast(Path, args.answer_provenance).resolve(),
        model_tag,
    )
    formal: FormalData = load_formal_data(
        formal_dir,
        model_tag,
        retrievals,
        gates,
        fixed_sources,
        provenance,
    )
    metrics: MetricData = load_metric_data(
        cast(Path, args.loading_metrics_long).resolve(),
        cast(Path, args.answer_metrics_long).resolve(),
        cast(Path, args.answer_summary).resolve(),
        model_tag,
        formal,
    )
    significance: JsonObject = load_significance(
        cast(Path, args.significance).resolve(),
        model_tag,
    )
    build_pack(
        output_dir,
        args,
        model_tag,
        retrieval_rows,
        router_payload,
        gating_rows,
        formal,
        metrics,
        significance,
    )
    print(
        canonical_json(
            {
                "event": "k2_public_pack_exported",
                "model": model_tag,
                "output_dir": str(output_dir),
                "manifest_sha256": sha256_file(
                    output_dir / "manifest.json"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
