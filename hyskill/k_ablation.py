"""Typed validation and metric helpers for the K-imagination ablation."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from typing import TypedDict, cast

from hyskill.generator import PASSAGE_TEMPLATE, SENTENCE_TEMPLATE, SKILL_TEMPLATE


K_VALUES: tuple[int, ...] = (1, 2, 4, 8, 10)
DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
    "bigcodebench",
)
FIXED_VARIANTS: tuple[str, ...] = (
    "naive_sentence",
    "naive_passage",
    "naive_skill",
    "hyskill",
    "two_stage",
)
ALL_VARIANTS: tuple[str, ...] = (*FIXED_VARIANTS, "routed")
METRIC_DEPTHS: tuple[int, ...] = (1, 5, 10, 50)
VAL_FRACTION: float = 0.2
VAL_SEED: int = 0
EXPECTED_TOTAL_ROWS: int = 3970
EXPECTED_UNIQUE_QUERIES: int = 3968
EXPECTED_CACHE_FILES_PER_SAMPLE: int = 11904
EXPECTED_TOP_K: int = 50
TEMPLATE_TEXTS: dict[str, str] = {
    "sentence": SENTENCE_TEMPLATE,
    "passage": PASSAGE_TEMPLATE,
    "skill": SKILL_TEMPLATE,
}
VARIANT_RETRIEVERS: dict[str, str] = {
    "naive_sentence": "naive_hyde",
    "naive_passage": "naive_hyde",
    "naive_skill": "naive_hyde",
    "hyskill": "hyskill",
    "two_stage": "two_stage",
}


class RetrievedItem(TypedDict):
    """One ranked skill in an SR-Agents retrieval record."""

    skill_id: str
    score: float


class RetrievalRecord(TypedDict):
    """One query-level retrieval record."""

    instance_id: str
    gold_skill_ids: list[str]
    retrieved: list[RetrievedItem]


class RetrievalPayload(TypedDict):
    """Normalized retrieval result payload."""

    metadata: dict[str, object]
    metrics: dict[str, float]
    results: list[RetrievalRecord]


class KResultSpec(TypedDict):
    """Identity fields that make one K-ablation result unambiguous."""

    tag: str
    model: str
    k_samples: int
    domain: str
    variant: str
    encoder: str
    source_revision: str


class CacheStamp(TypedDict):
    """Content identity of the generation cache used by one result."""

    manifest_path: str
    manifest_sha256: str
    artifact_sha256: str
    model_revision: str
    generation_code_commit: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(value.encode()).hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    """Load a JSON object and reject non-object roots."""

    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: path={path}")
    return cast(dict[str, object], value)


def require_object(value: object, context: str) -> dict[str, object]:
    """Return a typed JSON object or raise a contextual error."""

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected an object: context={context}, actual={type(value).__name__}"
        )
    return cast(dict[str, object], value)


def require_list(value: object, context: str) -> list[object]:
    """Return a typed JSON list or raise a contextual error."""

    if not isinstance(value, list):
        raise TypeError(
            f"Expected an array: context={context}, actual={type(value).__name__}"
        )
    return cast(list[object], value)


def require_string(value: object, context: str) -> str:
    """Return a non-empty string or raise a contextual error."""

    if not isinstance(value, str) or not value:
        raise TypeError(f"Expected a non-empty string: context={context}, value={value!r}")
    return value


def require_float(value: object, context: str) -> float:
    """Return a finite numeric value or raise a contextual error."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Expected a number: context={context}, value={value!r}")
    numeric_value: float = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"Expected a finite number: context={context}, value={value!r}")
    return numeric_value


def require_integer(value: object, context: str) -> int:
    """Return an integer or raise a contextual error."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Expected an integer: context={context}, value={value!r}")
    return value


def require_boolean(value: object, context: str) -> bool:
    """Return a Boolean or raise a contextual error."""

    if not isinstance(value, bool):
        raise TypeError(f"Expected a Boolean: context={context}, value={value!r}")
    return value


def require_sha256(value: object, context: str) -> str:
    """Return a lowercase SHA-256 digest or raise a contextual error."""

    digest: str = require_string(value, context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Expected a lowercase SHA-256 digest: context={context}, value={digest!r}")
    return digest


def parse_retrieval_payload(value: dict[str, object], context: str) -> RetrievalPayload:
    """Normalize a raw SR-Agents retrieval payload into strict typed records."""

    metadata: dict[str, object] = require_object(value.get("metadata"), f"{context}.metadata")
    raw_metrics: dict[str, object] = require_object(value.get("metrics"), f"{context}.metrics")
    raw_results: list[object] = require_list(value.get("results"), f"{context}.results")
    metrics: dict[str, float] = {
        key: require_float(metric_value, f"{context}.metrics.{key}")
        for key, metric_value in raw_metrics.items()
    }
    results: list[RetrievalRecord] = []
    for record_index, raw_record in enumerate(raw_results):
        record_context: str = f"{context}.results[{record_index}]"
        record: dict[str, object] = require_object(raw_record, record_context)
        instance_id: str = require_string(record.get("instance_id"), f"{record_context}.instance_id")
        raw_gold: list[object] = require_list(
            record.get("gold_skill_ids"), f"{record_context}.gold_skill_ids"
        )
        gold_skill_ids: list[str] = [
            require_string(skill_id, f"{record_context}.gold_skill_ids[{index}]")
            for index, skill_id in enumerate(raw_gold)
        ]
        raw_retrieved: list[object] = require_list(
            record.get("retrieved"), f"{record_context}.retrieved"
        )
        retrieved: list[RetrievedItem] = []
        for rank, raw_item in enumerate(raw_retrieved, start=1):
            item_context: str = f"{record_context}.retrieved[{rank - 1}]"
            item: dict[str, object] = require_object(raw_item, item_context)
            retrieved.append(
                {
                    "skill_id": require_string(item.get("skill_id"), f"{item_context}.skill_id"),
                    "score": require_float(item.get("score"), f"{item_context}.score"),
                }
            )
        results.append(
            {
                "instance_id": instance_id,
                "gold_skill_ids": gold_skill_ids,
                "retrieved": retrieved,
            }
        )
    return {"metadata": metadata, "metrics": metrics, "results": results}


def load_expected_gold(instances_path: Path, expected_domain: str) -> dict[str, list[str]]:
    """Load the exact evaluated instance IDs and gold skills for one domain."""

    value: object = json.loads(instances_path.read_text(encoding="utf-8"))
    raw_instances: list[object] = require_list(value, f"instances:{instances_path}")
    expected: dict[str, list[str]] = {}
    for row_index, raw_instance in enumerate(raw_instances):
        context: str = f"instances:{instances_path}[{row_index}]"
        instance: dict[str, object] = require_object(raw_instance, context)
        domain: str = require_string(instance.get("dataset"), f"{context}.dataset")
        if domain != expected_domain:
            raise ValueError(
                f"Instance domain mismatch: path={instances_path}, "
                f"expected={expected_domain}, actual={domain}, row={row_index}"
            )
        raw_gold: list[object] = require_list(
            instance.get("skill_annotations", []), f"{context}.skill_annotations"
        )
        if not raw_gold:
            continue
        instance_id: str = require_string(instance.get("instance_id"), f"{context}.instance_id")
        if instance_id in expected:
            raise ValueError(
                f"Duplicate instance ID in source data: path={instances_path}, "
                f"instance_id={instance_id}"
            )
        expected[instance_id] = [
            require_string(skill_id, f"{context}.skill_annotations[{index}]")
            for index, skill_id in enumerate(raw_gold)
        ]
    if not expected:
        raise ValueError(f"No annotated instances found: path={instances_path}")
    return expected


def metric_for_record(record: RetrievalRecord, metric_name: str) -> float:
    """Compute one retrieval metric for one query."""

    metric_family, separator, raw_depth = metric_name.partition("@")
    if separator != "@" or not raw_depth.isdigit():
        raise ValueError(f"Unsupported retrieval metric: metric={metric_name}")
    depth: int = int(raw_depth)
    if depth <= 0:
        raise ValueError(f"Metric depth must be positive: metric={metric_name}")
    gold: set[str] = set(record["gold_skill_ids"])
    ranked: list[str] = [item["skill_id"] for item in record["retrieved"][:depth]]
    if metric_family == "Recall":
        return len(set(ranked) & gold) / len(gold) if gold else 0.0
    if metric_family == "nDCG":
        dcg: float = sum(
            1.0 / math.log2(rank + 2)
            for rank, skill_id in enumerate(ranked)
            if skill_id in gold
        )
        idcg: float = sum(
            1.0 / math.log2(rank + 2) for rank in range(min(len(gold), depth))
        )
        return dcg / idcg if idcg > 0.0 else 0.0
    raise ValueError(f"Unsupported retrieval metric family: metric={metric_name}")


def compute_metrics(
    records: list[RetrievalRecord], selected_ids: frozenset[str]
) -> dict[str, float]:
    """Compute all protocol metrics on an explicit instance subset."""

    selected_records: list[RetrievalRecord] = [
        record for record in records if record["instance_id"] in selected_ids
    ]
    if len(selected_records) != len(selected_ids):
        present_ids: set[str] = {record["instance_id"] for record in selected_records}
        missing_ids: list[str] = sorted(selected_ids - present_ids)
        raise ValueError(
            f"Metric subset is missing records: expected={len(selected_ids)}, "
            f"actual={len(selected_records)}, missing_sample={missing_ids[:5]}"
        )
    metrics: dict[str, float] = {}
    for depth in METRIC_DEPTHS:
        for family in ("Recall", "nDCG"):
            metric_name: str = f"{family}@{depth}"
            values: list[float] = [
                metric_for_record(record, metric_name) for record in selected_records
            ]
            metrics[metric_name] = sum(values) / len(values)
    return metrics


def validation_ids(instance_ids: frozenset[str]) -> frozenset[str]:
    """Return the protocol's deterministic 20% validation split."""

    sorted_ids: list[str] = sorted(instance_ids)
    validation_count: int = max(1, int(len(sorted_ids) * VAL_FRACTION))
    return frozenset(random.Random(VAL_SEED).sample(sorted_ids, validation_count))


def instance_ids_sha256(instance_ids: frozenset[str]) -> str:
    """Return a stable digest for an explicit set of instance IDs."""

    encoded: str = json.dumps(sorted(instance_ids), ensure_ascii=False, separators=(",", ":"))
    return sha256_text(encoded)


def expected_router_pick(scores: dict[str, float]) -> tuple[str, bool]:
    """Return the protocol winner and whether validation was degenerate."""

    if set(scores) != set(FIXED_VARIANTS):
        raise ValueError(
            f"Router scores must cover every fixed variant: actual={sorted(scores)}"
        )
    degenerate: bool = max(scores.values()) <= 0.0
    if degenerate:
        return "naive_skill", True
    return max(FIXED_VARIANTS, key=lambda variant: scores[variant]), False


def _validate_result_core(
    payload: RetrievalPayload,
    expected_gold: dict[str, list[str]],
    spec: KResultSpec,
) -> None:
    metadata: dict[str, object] = payload["metadata"]
    if metadata.get("dataset") != spec["domain"]:
        raise ValueError(
            f"Result dataset mismatch: expected={spec['domain']}, "
            f"actual={metadata.get('dataset')!r}"
        )
    if metadata.get("top_k") != EXPECTED_TOP_K:
        raise ValueError(
            f"Result top_k must be {EXPECTED_TOP_K}: actual={metadata.get('top_k')!r}"
        )
    if metadata.get("n_queries") != len(expected_gold):
        raise ValueError(
            f"Result metadata query count mismatch: expected={len(expected_gold)}, "
            f"actual={metadata.get('n_queries')!r}"
        )
    records: list[RetrievalRecord] = payload["results"]
    by_id: dict[str, RetrievalRecord] = {}
    for record in records:
        instance_id: str = record["instance_id"]
        if instance_id in by_id:
            raise ValueError(f"Duplicate result instance ID: instance_id={instance_id}")
        by_id[instance_id] = record
        if len(record["retrieved"]) != EXPECTED_TOP_K:
            raise ValueError(
                f"Retrieved list must contain {EXPECTED_TOP_K} skills: instance_id={instance_id}, "
                f"actual={len(record['retrieved'])}"
            )
        ranked_ids: list[str] = [item["skill_id"] for item in record["retrieved"]]
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError(f"Retrieved list contains duplicate skills: instance_id={instance_id}")
    expected_ids: set[str] = set(expected_gold)
    actual_ids: set[str] = set(by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Result instance coverage mismatch: expected={len(expected_ids)}, "
            f"actual={len(actual_ids)}, missing_sample={sorted(expected_ids - actual_ids)[:5]}, "
            f"extra_sample={sorted(actual_ids - expected_ids)[:5]}"
        )
    for instance_id, expected_skills in expected_gold.items():
        actual_skills: list[str] = by_id[instance_id]["gold_skill_ids"]
        if actual_skills != expected_skills:
            raise ValueError(
                f"Gold skills mismatch: instance_id={instance_id}, "
                f"expected={expected_skills}, actual={actual_skills}"
            )
    recomputed: dict[str, float] = compute_metrics(records, frozenset(expected_ids))
    for metric_name, expected_value in recomputed.items():
        actual_value: float | None = payload["metrics"].get(metric_name)
        if actual_value is None:
            raise ValueError(f"Stored metrics are missing a required key: metric={metric_name}")
        if abs(actual_value - expected_value) > 1e-9:
            raise ValueError(
                f"Stored metric mismatch: metric={metric_name}, "
                f"stored={actual_value}, recomputed={expected_value}"
            )


def validate_raw_result(
    value: dict[str, object], instances_path: Path, spec: KResultSpec
) -> RetrievalPayload:
    """Validate an unstamped SR-Agents result against source instances."""

    payload: RetrievalPayload = parse_retrieval_payload(value, "retrieval")
    expected_gold: dict[str, list[str]] = load_expected_gold(instances_path, spec["domain"])
    _validate_result_core(payload, expected_gold, spec)
    return payload


def load_cache_stamp(
    cache_manifest_path: Path,
    instances_path: Path,
    spec: KResultSpec,
) -> CacheStamp:
    """Validate a cache manifest and return its immutable identity fields."""

    manifest: dict[str, object] = load_json_object(cache_manifest_path)
    expected_fields: dict[str, object] = {
        "schema_version": 1,
        "artifact": "complete_imagination_prefix_cache",
        "tag": spec["tag"],
        "model": spec["model"],
        "k_samples": spec["k_samples"],
        "temperature": 0.7,
        "rows": EXPECTED_TOTAL_ROWS,
        "verified_output_rows": EXPECTED_TOTAL_ROWS,
        "unique_queries": EXPECTED_UNIQUE_QUERIES,
        "verified_cache_files": EXPECTED_CACHE_FILES_PER_SAMPLE * spec["k_samples"],
    }
    for field, expected_value in expected_fields.items():
        actual_value: object = manifest.get(field)
        if actual_value != expected_value:
            raise ValueError(
                f"Cache manifest mismatch: path={cache_manifest_path}, field={field}, "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    templates: dict[str, object] = require_object(
        manifest.get("templates"), f"cache_manifest:{cache_manifest_path}.templates"
    )
    if set(templates) != set(TEMPLATE_TEXTS):
        raise ValueError(
            f"Cache manifest template set mismatch: path={cache_manifest_path}, "
            f"expected={sorted(TEMPLATE_TEXTS)}, actual={sorted(templates)}"
        )
    for template_name, template_text in TEMPLATE_TEXTS.items():
        template_entry: dict[str, object] = require_object(
            templates.get(template_name),
            f"cache_manifest:{cache_manifest_path}.templates.{template_name}",
        )
        actual_template_hash: str = require_sha256(
            template_entry.get("sha256"),
            f"cache_manifest:{cache_manifest_path}.templates.{template_name}.sha256",
        )
        expected_template_hash: str = sha256_text(template_text)
        if actual_template_hash != expected_template_hash:
            raise ValueError(
                f"Cache template hash mismatch: path={cache_manifest_path}, "
                f"template={template_name}, expected={expected_template_hash}, "
                f"actual={actual_template_hash}"
            )
    instance_entries: dict[str, object] = require_object(
        manifest.get("instances"), f"cache_manifest:{cache_manifest_path}.instances"
    )
    if set(instance_entries) != set(DOMAINS):
        raise ValueError(
            f"Cache manifest domain set mismatch: path={cache_manifest_path}, "
            f"expected={sorted(DOMAINS)}, actual={sorted(instance_entries)}"
        )
    domain_entry: dict[str, object] = require_object(
        instance_entries.get(spec["domain"]),
        f"cache_manifest:{cache_manifest_path}.instances.{spec['domain']}",
    )
    manifest_instance_hash: str = require_sha256(
        domain_entry.get("sha256"),
        f"cache_manifest:{cache_manifest_path}.instances.{spec['domain']}.sha256",
    )
    actual_instance_hash: str = sha256_file(instances_path)
    if manifest_instance_hash != actual_instance_hash:
        raise ValueError(
            f"Instance file hash mismatch: manifest={cache_manifest_path}, "
            f"instances={instances_path}, expected={manifest_instance_hash}, "
            f"actual={actual_instance_hash}"
        )
    expected_gold: dict[str, list[str]] = load_expected_gold(instances_path, spec["domain"])
    manifest_rows: int = require_integer(
        domain_entry.get("rows"),
        f"cache_manifest:{cache_manifest_path}.instances.{spec['domain']}.rows",
    )
    if manifest_rows != len(expected_gold):
        raise ValueError(
            f"Cache manifest domain row count mismatch: path={cache_manifest_path}, "
            f"domain={spec['domain']}, expected={len(expected_gold)}, actual={manifest_rows}"
        )
    output: dict[str, object] = require_object(
        manifest.get("output"), f"cache_manifest:{cache_manifest_path}.output"
    )
    artifact_sha256: str = require_sha256(
        output.get("sha256"), f"cache_manifest:{cache_manifest_path}.output.sha256"
    )
    require_string(output.get("path"), f"cache_manifest:{cache_manifest_path}.output.path")
    artifact_bytes: int = require_integer(
        output.get("bytes"), f"cache_manifest:{cache_manifest_path}.output.bytes"
    )
    if artifact_bytes <= 0:
        raise ValueError(
            f"Cache artifact byte count must be positive: path={cache_manifest_path}, "
            f"actual={artifact_bytes}"
        )
    model_revision: str = require_string(
        manifest.get("model_revision"),
        f"cache_manifest:{cache_manifest_path}.model_revision",
    )
    generation_code_commit: str = require_string(
        manifest.get("generation_code_commit"),
        f"cache_manifest:{cache_manifest_path}.generation_code_commit",
    )
    return {
        "manifest_path": str(cache_manifest_path),
        "manifest_sha256": sha256_file(cache_manifest_path),
        "artifact_sha256": artifact_sha256,
        "model_revision": model_revision,
        "generation_code_commit": generation_code_commit,
    }


def validate_cache_artifact(cache_manifest_path: Path, repository_root: Path) -> Path:
    """Verify the exported cache artifact bytes and JSONL row count once."""

    manifest: dict[str, object] = load_json_object(cache_manifest_path)
    output: dict[str, object] = require_object(
        manifest.get("output"), f"cache_manifest:{cache_manifest_path}.output"
    )
    relative_path: Path = Path(
        require_string(output.get("path"), f"cache_manifest:{cache_manifest_path}.output.path")
    )
    artifact_path: Path = (repository_root / relative_path).resolve()
    resolved_root: Path = repository_root.resolve()
    if not artifact_path.is_relative_to(resolved_root):
        raise ValueError(
            f"Cache artifact path escapes the repository: manifest={cache_manifest_path}, "
            f"artifact={artifact_path}, repository={resolved_root}"
        )
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Cache artifact is missing: manifest={cache_manifest_path}, artifact={artifact_path}"
        )
    expected_bytes: int = require_integer(
        output.get("bytes"), f"cache_manifest:{cache_manifest_path}.output.bytes"
    )
    actual_bytes: int = artifact_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Cache artifact size mismatch: artifact={artifact_path}, "
            f"expected={expected_bytes}, actual={actual_bytes}"
        )
    expected_digest: str = require_sha256(
        output.get("sha256"), f"cache_manifest:{cache_manifest_path}.output.sha256"
    )
    actual_digest: str = sha256_file(artifact_path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"Cache artifact digest mismatch: artifact={artifact_path}, "
            f"expected={expected_digest}, actual={actual_digest}"
        )
    row_count: int = 0
    with gzip.open(artifact_path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            parsed: object = json.loads(line)
            if not isinstance(parsed, dict):
                raise TypeError(
                    f"Cache artifact row must be an object: artifact={artifact_path}, "
                    f"line={line_number}"
                )
            row_count += 1
    if row_count != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Cache artifact row count mismatch: artifact={artifact_path}, "
            f"expected={EXPECTED_TOTAL_ROWS}, actual={row_count}"
        )
    return artifact_path


def finalize_result_payload(
    value: dict[str, object],
    instances_path: Path,
    cache_manifest_path: Path,
    spec: KResultSpec,
) -> dict[str, object]:
    """Return a validated copy stamped with immutable K-ablation metadata."""

    validate_raw_result(value, instances_path, spec)
    cache_stamp: CacheStamp = load_cache_stamp(cache_manifest_path, instances_path, spec)
    finalized: dict[str, object] = copy.deepcopy(value)
    metadata: dict[str, object] = require_object(finalized.get("metadata"), "retrieval.metadata")
    metadata["k_ablation"] = {
        "schema_version": 1,
        **spec,
        "cache": cache_stamp,
    }
    return finalized


def validate_final_result(
    value: dict[str, object],
    instances_path: Path,
    cache_manifest_path: Path,
    spec: KResultSpec,
) -> RetrievalPayload:
    """Validate a finalized result including its immutable K metadata stamp."""

    payload: RetrievalPayload = validate_raw_result(value, instances_path, spec)
    expected_gold: dict[str, list[str]] = load_expected_gold(
        instances_path, spec["domain"]
    )
    stamp: dict[str, object] = require_object(
        payload["metadata"].get("k_ablation"), "retrieval.metadata.k_ablation"
    )
    expected_stamp: dict[str, object] = {
        "schema_version": 1,
        **spec,
        "cache": load_cache_stamp(cache_manifest_path, instances_path, spec),
    }
    if stamp != expected_stamp:
        raise ValueError(
            f"K-ablation metadata mismatch: expected={expected_stamp!r}, actual={stamp!r}"
        )
    variant: str = spec["variant"]
    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unsupported K-ablation variant: variant={variant}")
    if variant == "routed":
        router: dict[str, object] = require_object(
            payload["metadata"].get("router"), "retrieval.metadata.router"
        )
        pick: str = require_string(router.get("pick"), "retrieval.metadata.router.pick")
        if pick not in FIXED_VARIANTS:
            raise ValueError(f"Router selected an unsupported variant: pick={pick}")
        if router.get("schema_version") != 1:
            raise ValueError(
                f"Router schema version mismatch: actual={router.get('schema_version')!r}"
            )
        if router.get("validation_metric") != "nDCG@10":
            raise ValueError(
                f"Router validation metric mismatch: actual={router.get('validation_metric')!r}"
            )
        if require_float(router.get("validation_fraction"), "retrieval.metadata.router.validation_fraction") != VAL_FRACTION:
            raise ValueError(
                f"Router validation fraction mismatch: actual={router.get('validation_fraction')!r}"
            )
        if require_integer(router.get("seed"), "retrieval.metadata.router.seed") != VAL_SEED:
            raise ValueError(f"Router seed mismatch: actual={router.get('seed')!r}")
        expected_ids: frozenset[str] = frozenset(expected_gold)
        expected_validation_ids: frozenset[str] = validation_ids(expected_ids)
        if require_integer(
            router.get("n_validation"), "retrieval.metadata.router.n_validation"
        ) != len(expected_validation_ids):
            raise ValueError(
                f"Router validation count mismatch: expected={len(expected_validation_ids)}, "
                f"actual={router.get('n_validation')!r}"
            )
        expected_ids_digest: str = instance_ids_sha256(expected_validation_ids)
        if require_sha256(
            router.get("validation_ids_sha256"),
            "retrieval.metadata.router.validation_ids_sha256",
        ) != expected_ids_digest:
            raise ValueError(
                f"Router validation split digest mismatch: expected={expected_ids_digest}, "
                f"actual={router.get('validation_ids_sha256')!r}"
            )
        raw_scores: dict[str, object] = require_object(
            router.get("validation_scores"), "retrieval.metadata.router.validation_scores"
        )
        scores: dict[str, float] = {
            score_variant: require_float(
                score_value,
                f"retrieval.metadata.router.validation_scores.{score_variant}",
            )
            for score_variant, score_value in raw_scores.items()
        }
        expected_pick, expected_degenerate = expected_router_pick(scores)
        if pick != expected_pick:
            raise ValueError(
                f"Router pick does not match validation scores: expected={expected_pick}, "
                f"actual={pick}, scores={scores}"
            )
        if require_boolean(
            router.get("degenerate"), "retrieval.metadata.router.degenerate"
        ) != expected_degenerate:
            raise ValueError(
                f"Router degenerate flag mismatch: expected={expected_degenerate}, "
                f"actual={router.get('degenerate')!r}"
            )
        source_result: dict[str, object] = require_object(
            router.get("source_result"), "retrieval.metadata.router.source_result"
        )
        require_string(source_result.get("path"), "retrieval.metadata.router.source_result.path")
        require_sha256(
            source_result.get("sha256"), "retrieval.metadata.router.source_result.sha256"
        )
        expected_retriever: str = VARIANT_RETRIEVERS[pick]
    else:
        expected_retriever = VARIANT_RETRIEVERS[variant]
    actual_retriever: object = payload["metadata"].get("retriever")
    if actual_retriever != expected_retriever:
        raise ValueError(
            f"Result retriever mismatch: variant={variant}, expected={expected_retriever}, "
            f"actual={actual_retriever!r}"
        )
    return payload
