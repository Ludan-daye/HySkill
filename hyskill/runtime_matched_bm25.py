"""Deterministic native-BM25 artifacts for runtime-matched baselines."""

from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict, cast

from hyskill.runtime_matched_execution import (
    JsonLike,
    JsonObject,
    JsonValue,
    load_json_file,
    sha256_file,
    sha256_json,
    write_json_atomic,
)


BM25_SCHEMA_VERSION: str = "runtime-matched-bm25-v1"
BM25_MANIFEST_SCHEMA_VERSION: str = "runtime-matched-bm25-manifest-v1"
BM25_VALIDATION_SCHEMA_VERSION: str = "runtime-matched-bm25-validation-v1"
FROZEN_TOP_K: int = 50
FROZEN_SRAGENTS_REVISION: str = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
FROZEN_RETRIEVER_NAME: str = "fast_bm25"
FROZEN_BM25_BACKEND: str = "bm25s"
FROZEN_BM25S_VERSION: str = "0.3.9"
FROZEN_TOKENIZATION: str = "lower_whitespace_split"
FROZEN_SCORING_METHOD: str = "robertson"

class RuntimeMatchedBM25Error(ValueError):
    """Raised when a frozen BM25 input or artifact violates its contract."""


class NativeRetriever(Protocol):
    """SR-Agents native retriever interface."""

    def build_index(
        self,
        corpus_ids: list[str],
        corpus_texts: list[str],
    ) -> None:
        """Index one ordered corpus."""

    def retrieve(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[list[tuple[str, float]]]:
        """Return one ordered ranking per query."""


class RetrieverFactory(Protocol):
    """Native BM25 constructor interface."""

    def __call__(self) -> NativeRetriever:
        """Construct a fresh native BM25 retriever."""


class NativeBM25Runtime(TypedDict):
    """Pinned query, corpus, and legacy fast-BM25 runtime."""

    build_query: Callable[[JsonObject], str]
    skill_text: Callable[[JsonObject], str]
    create_retriever: RetrieverFactory
    retriever_name: str
    retriever_backend: str
    bm25s_version: str
    tokenization: str
    scoring_method: str
    revision: str
    source_root: str


class RetrievedCandidate(TypedDict):
    """One BM25 candidate in standard retrieval order."""

    skill_id: str
    score: float


class BM25Record(TypedDict):
    """One deterministic retrieval record."""

    instance_id: str
    gold_skill_ids: list[str]
    retrieved: list[RetrievedCandidate]


class BM25Artifact(TypedDict):
    """Model-independent BM25 top-50 artifact."""

    schema_version: str
    metadata: JsonObject
    results: list[BM25Record]


class BM25Manifest(TypedDict):
    """Hashes and counts binding one BM25 artifact to frozen inputs."""

    schema_version: str
    domain: str
    top_k: int
    corpus_sha256: str
    instances_sha256: str
    artifact_sha256: str
    ordered_candidate_ids_sha256: str
    retriever: str
    retriever_backend: str
    bm25s_version: str
    tokenization: str
    scoring_method: str
    sragents_revision: str
    sragents_source_root: str
    corpus_size: int
    instance_count: int
    code_files: list[JsonObject]


class BM25ValidationReport(TypedDict):
    """Successful exact validation against the legacy candidate source."""

    schema_version: str
    valid: bool
    domain: str
    top_k: int
    instance_count: int
    corpus_size: int
    artifact_sha256: str
    manifest_sha256: str
    legacy_source_sha256: str
    ordered_candidate_ids_sha256: str
    legacy_ordered_candidate_ids_sha256: str
    coverage: JsonObject
    duplicate_candidate_ids: int
    order_mismatches: int


def require_object(value: JsonValue, context: str) -> JsonObject:
    """Return a JSON object or raise with context."""

    if not isinstance(value, dict):
        raise RuntimeMatchedBM25Error(
            f"Expected JSON object: context={context}, "
            f"value_type={type(value).__name__}"
        )
    return value


def require_list(value: JsonValue, context: str) -> list[JsonValue]:
    """Return a JSON list or raise with context."""

    if not isinstance(value, list):
        raise RuntimeMatchedBM25Error(
            f"Expected JSON list: context={context}, "
            f"value_type={type(value).__name__}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return a non-empty string or raise with context."""

    if not isinstance(value, str) or not value:
        raise RuntimeMatchedBM25Error(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def object_rows(values: Sequence[JsonValue], context: str) -> list[JsonObject]:
    """Validate a sequence of JSON object rows."""

    return [
        require_object(value, f"{context}[{index}]")
        for index, value in enumerate(values)
    ]


def unique_rows_by_id(
    rows: Sequence[JsonObject],
    id_field: str,
    context: str,
) -> dict[str, JsonObject]:
    """Index rows by a required unique string identifier."""

    output: dict[str, JsonObject] = {}
    for index, row in enumerate(rows):
        row_id: str = require_string(
            row.get(id_field),
            f"{context}[{index}].{id_field}",
        )
        if row_id in output:
            raise RuntimeMatchedBM25Error(
                f"Duplicate identifier: context={context}, "
                f"field={id_field}, value={row_id}"
            )
        output[row_id] = row
    return output


def read_json(path: Path) -> JsonValue:
    """Read one UTF-8 JSON file with precise parse errors."""

    try:
        return load_json_file(path, "BM25")
    except ValueError as error:
        raise RuntimeMatchedBM25Error(
            f"Unable to load BM25 JSON: path={path}, error={error}"
        ) from error


def write_json(path: Path, value: object) -> None:
    """Atomically write stable, human-readable JSON."""

    write_json_atomic(path, cast(JsonLike, value))


def git_revision(checkout: Path) -> str:
    """Read the exact HEAD revision of a required Git checkout."""

    if not checkout.is_dir():
        raise NotADirectoryError(
            f"SR-Agents checkout does not exist: path={checkout}"
        )
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to read SR-Agents revision: "
            f"path={checkout}, returncode={result.returncode}, "
            f"stderr={result.stderr.strip()}"
        )
    return result.stdout.strip()


def require_module_under(module: ModuleType, source_root: Path) -> None:
    """Reject an imported module from outside its required source root."""

    raw_path: str | None = cast(str | None, getattr(module, "__file__", None))
    if raw_path is None:
        raise RuntimeError(
            f"Imported module has no source path: module={module.__name__}"
        )
    module_path: Path = Path(raw_path).resolve()
    if not module_path.is_relative_to(source_root):
        raise RuntimeError(
            "Imported module is not from the pinned source root: "
            f"module={module.__name__}, module_path={module_path}, "
            f"source_root={source_root}"
        )


def load_native_bm25_runtime(
    checkout: Path,
    expected_revision: str,
) -> NativeBM25Runtime:
    """Load pinned SR query/corpus helpers and the legacy fast-BM25 backend."""

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
    retrieve_module: ModuleType = importlib.import_module(
        "sragents.cli.retrieve"
    )
    corpus_module: ModuleType = importlib.import_module("sragents.corpus")
    for module in (retrieve_module, corpus_module):
        require_module_under(module, source_root)
    project_root: Path = Path(__file__).resolve().parents[1]
    fast_bm25_module: ModuleType = importlib.import_module("hyskill.bm25")
    require_module_under(fast_bm25_module, project_root)
    try:
        importlib.import_module(FROZEN_BM25_BACKEND)
        bm25s_version: str = importlib.metadata.version(FROZEN_BM25_BACKEND)
    except (
        ModuleNotFoundError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        raise RuntimeError(
            "Legacy fast_bm25 requires bm25s and forbids rank_bm25 "
            "fallback: install the pinned project environment with "
            f"bm25s=={FROZEN_BM25S_VERSION}"
        ) from error
    if bm25s_version != FROZEN_BM25S_VERSION:
        raise RuntimeError(
            "Legacy fast_bm25 bm25s version mismatch: "
            f"expected={FROZEN_BM25S_VERSION}, actual={bm25s_version}"
        )
    return {
        "build_query": cast(
            Callable[[JsonObject], str],
            getattr(retrieve_module, "_build_query"),
        ),
        "skill_text": cast(
            Callable[[JsonObject], str],
            getattr(corpus_module, "skill_text"),
        ),
        "create_retriever": cast(
            RetrieverFactory,
            getattr(fast_bm25_module, "FastBM25Retriever"),
        ),
        "retriever_name": FROZEN_RETRIEVER_NAME,
        "retriever_backend": FROZEN_BM25_BACKEND,
        "bm25s_version": bm25s_version,
        "tokenization": FROZEN_TOKENIZATION,
        "scoring_method": FROZEN_SCORING_METHOD,
        "revision": observed_revision,
        "source_root": str(source_root),
    }


def validate_runtime_identity(runtime: NativeBM25Runtime) -> None:
    """Require the exact legacy fast-BM25 implementation identity."""

    expected_fields: tuple[tuple[str, str], ...] = (
        ("retriever_name", FROZEN_RETRIEVER_NAME),
        ("retriever_backend", FROZEN_BM25_BACKEND),
        ("bm25s_version", FROZEN_BM25S_VERSION),
        ("tokenization", FROZEN_TOKENIZATION),
        ("scoring_method", FROZEN_SCORING_METHOD),
        ("revision", FROZEN_SRAGENTS_REVISION),
    )
    mismatches: list[str] = [
        f"{field}:expected={expected!r},actual={runtime[field]!r}"
        for field, expected in expected_fields
        if runtime[field] != expected
    ]
    if mismatches:
        raise RuntimeMatchedBM25Error(
            "Legacy fast_bm25 runtime identity mismatch: "
            f"mismatches={mismatches}"
        )


def validate_retriever_backend(retriever: NativeRetriever) -> None:
    """Reject any FastBM25 fallback after index construction."""

    bm25_impl: object | None = getattr(retriever, "_bm25", None)
    observed_backend: object | None = getattr(
        bm25_impl,
        "_backend",
        None,
    )
    if observed_backend != FROZEN_BM25_BACKEND:
        raise RuntimeMatchedBM25Error(
            "Legacy fast_bm25 backend mismatch; fallback is forbidden: "
            f"expected={FROZEN_BM25_BACKEND!r}, "
            f"actual={observed_backend!r}"
        )


def validated_corpus(corpus_values: Sequence[JsonValue]) -> list[JsonObject]:
    """Validate corpus identity while preserving its frozen file order."""

    corpus: list[JsonObject] = object_rows(corpus_values, "corpus")
    unique_rows_by_id(corpus, "skill_id", "corpus")
    if len(corpus) < FROZEN_TOP_K:
        raise RuntimeMatchedBM25Error(
            "Corpus is too small for the frozen top-50 pool: "
            f"corpus_size={len(corpus)}, required={FROZEN_TOP_K}"
        )
    return corpus


def validated_instances(
    instance_values: Sequence[JsonValue],
    domain: str,
) -> list[JsonObject]:
    """Validate one complete domain instance file in its frozen order."""

    instances: list[JsonObject] = object_rows(instance_values, "instances")
    unique_rows_by_id(instances, "instance_id", "instances")
    for index, instance in enumerate(instances):
        observed_domain: str = require_string(
            instance.get("dataset"),
            f"instances[{index}].dataset",
        )
        if observed_domain != domain:
            raise RuntimeMatchedBM25Error(
                "Instance domain mismatch: "
                f"instance_id={instance.get('instance_id')!r}, "
                f"expected={domain}, actual={observed_domain}"
            )
        gold_values: list[JsonValue] = require_list(
            cast(JsonValue, instance.get("skill_annotations")),
            f"instances[{index}].skill_annotations",
        )
        if not gold_values:
            raise RuntimeMatchedBM25Error(
                "Instance has no frozen skill annotation: "
                f"instance_id={instance.get('instance_id')!r}"
            )
        for gold_index, gold_value in enumerate(gold_values):
            require_string(
                gold_value,
                f"instances[{index}].skill_annotations[{gold_index}]",
            )
    return instances


def validate_ranking(
    ranking: Sequence[tuple[str, float]],
    corpus_ids: frozenset[str],
    instance_id: str,
    top_k: int,
) -> list[RetrievedCandidate]:
    """Validate native output count, order identity, and score finiteness."""

    if len(ranking) != top_k:
        raise RuntimeMatchedBM25Error(
            "Native BM25 returned the wrong candidate count: "
            f"instance_id={instance_id}, expected={top_k}, actual={len(ranking)}"
        )
    output: list[RetrievedCandidate] = []
    seen: set[str] = set()
    for rank, (skill_id, score) in enumerate(ranking, start=1):
        if skill_id not in corpus_ids:
            raise RuntimeMatchedBM25Error(
                "Native BM25 returned an unknown skill: "
                f"instance_id={instance_id}, rank={rank}, skill_id={skill_id}"
            )
        if skill_id in seen:
            raise RuntimeMatchedBM25Error(
                "Native BM25 returned a duplicate skill: "
                f"instance_id={instance_id}, rank={rank}, skill_id={skill_id}"
            )
        if not isinstance(score, (int, float)):
            raise RuntimeMatchedBM25Error(
                "Native BM25 returned a non-numeric score: "
                f"instance_id={instance_id}, rank={rank}, score={score!r}"
            )
        numeric_score: float = float(score)
        if numeric_score != numeric_score or numeric_score in (
            float("inf"),
            float("-inf"),
        ):
            raise RuntimeMatchedBM25Error(
                "Native BM25 returned a non-finite score: "
                f"instance_id={instance_id}, rank={rank}, score={numeric_score}"
            )
        seen.add(skill_id)
        output.append({"skill_id": skill_id, "score": numeric_score})
    return output


def ordered_candidate_id_payload(
    records: Sequence[BM25Record],
) -> list[JsonObject]:
    """Return the model-visible ordered candidate identity for hashing."""

    return [
        {
            "instance_id": record["instance_id"],
            "ordered_candidate_ids": [
                candidate["skill_id"] for candidate in record["retrieved"]
            ],
        }
        for record in records
    ]


def build_bm25_artifact(
    instances: Sequence[JsonObject],
    corpus: Sequence[JsonObject],
    domain: str,
    runtime: NativeBM25Runtime,
    corpus_sha256: str,
    instances_sha256: str,
    top_k: int,
) -> BM25Artifact:
    """Build one deterministic legacy fast-BM25 artifact."""

    if top_k != FROZEN_TOP_K:
        raise RuntimeMatchedBM25Error(
            f"Runtime-matched BM25 requires top_k={FROZEN_TOP_K}: actual={top_k}"
        )
    validate_runtime_identity(runtime)
    corpus_ids: list[str] = [
        require_string(skill.get("skill_id"), f"corpus[{index}].skill_id")
        for index, skill in enumerate(corpus)
    ]
    corpus_texts: list[str] = [
        runtime["skill_text"](skill) for skill in corpus
    ]
    queries: list[str] = [
        runtime["build_query"](instance) for instance in instances
    ]
    retriever: NativeRetriever = runtime["create_retriever"]()
    retriever.build_index(corpus_ids, corpus_texts)
    validate_retriever_backend(retriever)
    raw_rankings: list[list[tuple[str, float]]] = retriever.retrieve(
        queries,
        top_k,
    )
    if len(raw_rankings) != len(instances):
        raise RuntimeMatchedBM25Error(
            "Native BM25 returned the wrong number of rankings: "
            f"expected={len(instances)}, actual={len(raw_rankings)}"
        )
    corpus_id_set: frozenset[str] = frozenset(corpus_ids)
    records: list[BM25Record] = []
    for instance, ranking in zip(instances, raw_rankings, strict=True):
        instance_id: str = require_string(
            instance.get("instance_id"),
            "instance.instance_id",
        )
        gold_values: list[JsonValue] = require_list(
            cast(JsonValue, instance.get("skill_annotations")),
            f"instance:{instance_id}.skill_annotations",
        )
        records.append(
            {
                "instance_id": instance_id,
                "gold_skill_ids": [
                    require_string(
                        value,
                        f"instance:{instance_id}.skill_annotations[{index}]",
                    )
                    for index, value in enumerate(gold_values)
                ],
                "retrieved": validate_ranking(
                    ranking,
                    corpus_id_set,
                    instance_id,
                    top_k,
                ),
            }
        )
    ordered_sha: str = sha256_json(ordered_candidate_id_payload(records))
    return {
        "schema_version": BM25_SCHEMA_VERSION,
        "metadata": {
            "dataset": domain,
            "retriever": runtime["retriever_name"],
            "retriever_backend": runtime["retriever_backend"],
            "bm25s_version": runtime["bm25s_version"],
            "tokenization": runtime["tokenization"],
            "scoring_method": runtime["scoring_method"],
            "top_k": top_k,
            "corpus_size": len(corpus),
            "n_queries": len(instances),
            "corpus_sha256": corpus_sha256,
            "instances_sha256": instances_sha256,
            "ordered_candidate_ids_sha256": ordered_sha,
            "sragents_revision": runtime["revision"],
        },
        "results": records,
    }


def code_file_digests(paths: Sequence[Path]) -> list[JsonObject]:
    """Return sorted code member hashes for the candidate manifest."""

    return [
        {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
        for path in sorted(paths, key=lambda value: str(value.resolve()))
    ]


def build_bm25_manifest(
    artifact: BM25Artifact,
    artifact_path: Path,
    corpus_path: Path,
    instances_path: Path,
    runtime: NativeBM25Runtime,
    code_files: Sequence[Path],
) -> BM25Manifest:
    """Build a manifest after the artifact has been atomically written."""

    metadata: JsonObject = artifact["metadata"]
    return {
        "schema_version": BM25_MANIFEST_SCHEMA_VERSION,
        "domain": require_string(metadata.get("dataset"), "metadata.dataset"),
        "top_k": cast(int, metadata["top_k"]),
        "corpus_sha256": sha256_file(corpus_path),
        "instances_sha256": sha256_file(instances_path),
        "artifact_sha256": sha256_file(artifact_path),
        "ordered_candidate_ids_sha256": require_string(
            metadata.get("ordered_candidate_ids_sha256"),
            "metadata.ordered_candidate_ids_sha256",
        ),
        "retriever": runtime["retriever_name"],
        "retriever_backend": runtime["retriever_backend"],
        "bm25s_version": runtime["bm25s_version"],
        "tokenization": runtime["tokenization"],
        "scoring_method": runtime["scoring_method"],
        "sragents_revision": runtime["revision"],
        "sragents_source_root": runtime["source_root"],
        "corpus_size": cast(int, metadata["corpus_size"]),
        "instance_count": cast(int, metadata["n_queries"]),
        "code_files": code_file_digests(code_files),
    }


def artifact_records(
    payload: JsonObject,
    context: str,
) -> list[BM25Record]:
    """Parse standard retrieval records from an artifact-like JSON object."""

    raw_results: list[JsonValue] = require_list(
        cast(JsonValue, payload.get("results")),
        f"{context}.results",
    )
    output: list[BM25Record] = []
    for index, raw_record in enumerate(raw_results):
        record: JsonObject = require_object(
            raw_record,
            f"{context}.results[{index}]",
        )
        instance_id: str = require_string(
            record.get("instance_id"),
            f"{context}.results[{index}].instance_id",
        )
        gold_values: list[JsonValue] = require_list(
            cast(JsonValue, record.get("gold_skill_ids")),
            f"{context}:{instance_id}.gold_skill_ids",
        )
        retrieved_values: list[JsonValue] = require_list(
            cast(JsonValue, record.get("retrieved")),
            f"{context}:{instance_id}.retrieved",
        )
        retrieved: list[RetrievedCandidate] = []
        for rank, raw_candidate in enumerate(retrieved_values, start=1):
            candidate: JsonObject = require_object(
                raw_candidate,
                f"{context}:{instance_id}.retrieved[{rank - 1}]",
            )
            raw_score: JsonValue | None = candidate.get("score")
            if not isinstance(raw_score, (int, float)):
                raise RuntimeMatchedBM25Error(
                    "Candidate score must be numeric: "
                    f"context={context}, instance_id={instance_id}, "
                    f"rank={rank}, score={raw_score!r}"
                )
            retrieved.append(
                {
                    "skill_id": require_string(
                        candidate.get("skill_id"),
                        f"{context}:{instance_id}.retrieved[{rank - 1}].skill_id",
                    ),
                    "score": float(raw_score),
                }
            )
        output.append(
            {
                "instance_id": instance_id,
                "gold_skill_ids": [
                    require_string(
                        value,
                        f"{context}:{instance_id}.gold_skill_ids[{gold_index}]",
                    )
                    for gold_index, value in enumerate(gold_values)
                ],
                "retrieved": retrieved,
            }
        )
    unique_rows_by_id(
        [cast(JsonObject, record) for record in output],
        "instance_id",
        f"{context}.results",
    )
    return output


def validate_bm25_artifact(
    artifact_payload: JsonObject,
    manifest_payload: JsonObject,
    legacy_payload: JsonObject,
    artifact_path: Path,
    manifest_path: Path,
    legacy_path: Path,
    corpus_path: Path,
    instances_path: Path,
    domain: str,
) -> BM25ValidationReport:
    """Validate exact coverage, top-50 identity, hashes, and legacy order."""

    if artifact_payload.get("schema_version") != BM25_SCHEMA_VERSION:
        raise RuntimeMatchedBM25Error(
            "BM25 artifact schema mismatch: "
            f"expected={BM25_SCHEMA_VERSION}, "
            f"actual={artifact_payload.get('schema_version')!r}"
        )
    if manifest_payload.get("schema_version") != BM25_MANIFEST_SCHEMA_VERSION:
        raise RuntimeMatchedBM25Error(
            "BM25 manifest schema mismatch: "
            f"expected={BM25_MANIFEST_SCHEMA_VERSION}, "
            f"actual={manifest_payload.get('schema_version')!r}"
        )
    corpus_rows: list[JsonObject] = validated_corpus(
        require_list(read_json(corpus_path), "corpus")
    )
    instance_rows: list[JsonObject] = validated_instances(
        require_list(read_json(instances_path), "instances"),
        domain,
    )
    expected_ids: list[str] = [
        require_string(instance.get("instance_id"), "instance.instance_id")
        for instance in instance_rows
    ]
    corpus_ids: frozenset[str] = frozenset(
        require_string(skill.get("skill_id"), "corpus.skill_id")
        for skill in corpus_rows
    )
    artifact_rows: list[BM25Record] = artifact_records(
        artifact_payload,
        "artifact",
    )
    legacy_rows: list[BM25Record] = artifact_records(legacy_payload, "legacy")
    legacy_metadata: JsonObject = require_object(
        cast(JsonValue, legacy_payload.get("metadata")),
        "legacy.metadata",
    )
    for field, expected in (
        ("dataset", domain),
        ("retriever", FROZEN_RETRIEVER_NAME),
        ("top_k", FROZEN_TOP_K),
        ("corpus_size", len(corpus_rows)),
        ("n_queries", len(instance_rows)),
    ):
        if legacy_metadata.get(field) != expected:
            raise RuntimeMatchedBM25Error(
                "Legacy candidate source identity mismatch: "
                f"field={field}, expected={expected!r}, "
                f"actual={legacy_metadata.get(field)!r}"
            )
    artifact_index: dict[str, BM25Record] = {
        record["instance_id"]: record for record in artifact_rows
    }
    legacy_index: dict[str, BM25Record] = {
        record["instance_id"]: record for record in legacy_rows
    }
    expected_set: set[str] = set(expected_ids)
    for name, observed in (
        ("artifact", set(artifact_index)),
        ("legacy", set(legacy_index)),
    ):
        if observed != expected_set:
            raise RuntimeMatchedBM25Error(
                f"{name} coverage mismatch: "
                f"missing={sorted(expected_set - observed)[:20]}, "
                f"unexpected={sorted(observed - expected_set)[:20]}"
            )
    duplicate_candidate_ids: int = 0
    order_mismatches: list[str] = []
    for instance in instance_rows:
        instance_id: str = require_string(
            instance.get("instance_id"),
            "instance.instance_id",
        )
        artifact_record: BM25Record = artifact_index[instance_id]
        legacy_record: BM25Record = legacy_index[instance_id]
        artifact_ids: list[str] = [
            candidate["skill_id"] for candidate in artifact_record["retrieved"]
        ]
        legacy_ids: list[str] = [
            candidate["skill_id"] for candidate in legacy_record["retrieved"]
        ][:FROZEN_TOP_K]
        duplicate_candidate_ids += len(artifact_ids) - len(set(artifact_ids))
        if len(artifact_ids) != FROZEN_TOP_K:
            raise RuntimeMatchedBM25Error(
                "Artifact candidate count mismatch: "
                f"instance_id={instance_id}, expected={FROZEN_TOP_K}, "
                f"actual={len(artifact_ids)}"
            )
        unknown_ids: list[str] = sorted(set(artifact_ids) - corpus_ids)
        if unknown_ids:
            raise RuntimeMatchedBM25Error(
                "Artifact contains candidates outside the frozen corpus: "
                f"instance_id={instance_id}, sample={unknown_ids[:20]}"
            )
        expected_gold: list[str] = [
            require_string(value, f"instance:{instance_id}.skill_annotations")
            for value in require_list(
                cast(JsonValue, instance.get("skill_annotations")),
                f"instance:{instance_id}.skill_annotations",
            )
        ]
        if artifact_record["gold_skill_ids"] != expected_gold:
            raise RuntimeMatchedBM25Error(
                "Artifact gold skills differ from frozen instances: "
                f"instance_id={instance_id}, expected={expected_gold}, "
                f"actual={artifact_record['gold_skill_ids']}"
            )
        if artifact_ids != legacy_ids:
            order_mismatches.append(instance_id)
    if duplicate_candidate_ids:
        raise RuntimeMatchedBM25Error(
            "Artifact contains duplicate candidate IDs: "
            f"duplicate_count={duplicate_candidate_ids}"
        )
    if order_mismatches:
        raise RuntimeMatchedBM25Error(
            "BM25 ordered candidate IDs differ from the legacy source: "
            f"mismatch_count={len(order_mismatches)}, "
            f"sample={order_mismatches[:20]}"
        )
    artifact_order_sha: str = sha256_json(
        ordered_candidate_id_payload(artifact_rows)
    )
    legacy_order_sha: str = sha256_json(
        [
            {
                "instance_id": instance_id,
                "ordered_candidate_ids": [
                    candidate["skill_id"]
                    for candidate in legacy_index[instance_id]["retrieved"][
                        :FROZEN_TOP_K
                    ]
                ],
            }
            for instance_id in expected_ids
        ]
    )
    expected_manifest_values: tuple[tuple[str, JsonValue], ...] = (
        ("domain", domain),
        ("top_k", FROZEN_TOP_K),
        ("corpus_sha256", sha256_file(corpus_path)),
        ("instances_sha256", sha256_file(instances_path)),
        ("artifact_sha256", sha256_file(artifact_path)),
        ("ordered_candidate_ids_sha256", artifact_order_sha),
        ("retriever", FROZEN_RETRIEVER_NAME),
        ("retriever_backend", FROZEN_BM25_BACKEND),
        ("bm25s_version", FROZEN_BM25S_VERSION),
        ("tokenization", FROZEN_TOKENIZATION),
        ("scoring_method", FROZEN_SCORING_METHOD),
        ("sragents_revision", FROZEN_SRAGENTS_REVISION),
        ("corpus_size", len(corpus_rows)),
        ("instance_count", len(instance_rows)),
    )
    for field, expected in expected_manifest_values:
        actual: JsonValue | None = manifest_payload.get(field)
        if actual != expected:
            raise RuntimeMatchedBM25Error(
                "BM25 manifest field mismatch: "
                f"field={field}, expected={expected!r}, actual={actual!r}"
            )
    metadata: JsonObject = require_object(
        cast(JsonValue, artifact_payload.get("metadata")),
        "artifact.metadata",
    )
    for field, expected in (
        ("dataset", domain),
        ("top_k", FROZEN_TOP_K),
        ("corpus_size", len(corpus_rows)),
        ("n_queries", len(instance_rows)),
        ("corpus_sha256", sha256_file(corpus_path)),
        ("instances_sha256", sha256_file(instances_path)),
        ("ordered_candidate_ids_sha256", artifact_order_sha),
        ("retriever", FROZEN_RETRIEVER_NAME),
        ("retriever_backend", FROZEN_BM25_BACKEND),
        ("bm25s_version", FROZEN_BM25S_VERSION),
        ("tokenization", FROZEN_TOKENIZATION),
        ("scoring_method", FROZEN_SCORING_METHOD),
        ("sragents_revision", FROZEN_SRAGENTS_REVISION),
    ):
        if metadata.get(field) != expected:
            raise RuntimeMatchedBM25Error(
                "BM25 artifact metadata mismatch: "
                f"field={field}, expected={expected!r}, "
                f"actual={metadata.get(field)!r}"
            )
    return {
        "schema_version": BM25_VALIDATION_SCHEMA_VERSION,
        "valid": True,
        "domain": domain,
        "top_k": FROZEN_TOP_K,
        "instance_count": len(instance_rows),
        "corpus_size": len(corpus_rows),
        "artifact_sha256": sha256_file(artifact_path),
        "manifest_sha256": sha256_file(manifest_path),
        "legacy_source_sha256": sha256_file(legacy_path),
        "ordered_candidate_ids_sha256": artifact_order_sha,
        "legacy_ordered_candidate_ids_sha256": legacy_order_sha,
        "coverage": {
            "expected": len(expected_ids),
            "observed": len(artifact_rows),
            "missing": 0,
            "unexpected": 0,
        },
        "duplicate_candidate_ids": duplicate_candidate_ids,
        "order_mismatches": 0,
    }
