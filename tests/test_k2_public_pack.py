import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

import pytest

from hyskill.downstream_reuse import JsonLike, JsonObject, canonical_json
from hyskill.k_ablation import (
    FIXED_VARIANTS,
    instance_ids_sha256,
    validation_ids,
)
from hyskill.loading_metrics import compute_loading_metrics
from scripts.export_k2_public_pack import (
    ProvenanceData,
    public_answer_provenance,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
    "bigcodebench",
)
RULE_DOMAINS: tuple[str, ...] = RETRIEVAL_DOMAINS[:4]
SELECT_ELIGIBLE_MODELS: frozenset[str] = frozenset(
    {
        "qwen3.5-4b-reference",
        "qwen35-9b",
        "glm4-9b",
        "llama31-8b",
        "mistral7b",
    }
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


class FixturePack(TypedDict):
    """All explicit inputs for one fixture exporter invocation."""

    model_tag: str
    formal_dir: Path
    retrieval_paths: dict[str, Path]
    router_paths: dict[str, Path]
    signals_paths: dict[str, Path]
    taus_paths: dict[str, Path]
    gated_paths: dict[str, Path]
    fixed_gated_paths: dict[str, Path]
    loading_metrics_path: Path
    answer_metrics_path: Path
    answer_summary_path: Path
    significance_path: Path
    answer_provenance_path: Path


def sha256_path(path: Path) -> str:
    """Return one fixture file digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: JsonLike) -> None:
    """Write compact deterministic fixture JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[JsonObject]) -> None:
    """Write canonical fixture JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[JsonObject]:
    """Read one plain fixture JSONL file."""

    return [
        cast(JsonObject, json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_gzip_rows(path: Path) -> list[JsonObject]:
    """Read public gzip JSONL rows."""

    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        return [
            cast(JsonObject, json.loads(line))
            for line in input_file
            if line.strip()
        ]


def instance_ids(domain: str) -> tuple[str, str]:
    """Return two deterministic fixture instance IDs."""

    return (f"{domain}_00000", f"{domain}_00001")


def router_metadata(domain: str) -> JsonObject:
    """Build one valid frozen router decision."""

    ids: frozenset[str] = frozenset(instance_ids(domain))
    selected_ids: frozenset[str] = validation_ids(ids)
    scores: JsonObject = {
        variant: 1.0 if variant == "naive_skill" else 0.5
        for variant in FIXED_VARIANTS
    }
    return {
        "schema_version": 1,
        "pick": "naive_skill",
        "validation_metric": "nDCG@10",
        "validation_scores": scores,
        "n_validation": len(selected_ids),
        "validation_fraction": 0.2,
        "seed": 0,
        "validation_ids_sha256": instance_ids_sha256(selected_ids),
        "degenerate": False,
        "source_result": {
            "path": f"/fixture/{domain}-naive_skill.json",
            "sha256": "1" * 64,
        },
    }


def k_stamp(
    model_tag: str,
    domain: str,
    variant: str,
) -> JsonObject:
    """Build one protocol-shaped K-ablation stamp."""

    return {
        "schema_version": 1,
        "tag": model_tag,
        "model": CACHE_MODEL_TAGS[model_tag],
        "k_samples": 2,
        "domain": domain,
        "variant": variant,
        "encoder": "fixture-encoder",
        "source_revision": "fixture-source",
        "cache": {
            "manifest_path": f"/fixture/{model_tag}-k2.manifest.json",
            "manifest_sha256": "2" * 64,
            "artifact_sha256": "3" * 64,
            "model_revision": "fixture-model-revision",
            "generation_code_commit": "fixture-code",
        },
    }


def retrieved_items() -> list[JsonObject]:
    """Return exactly 50 ordered fixture candidates."""

    return [
        {
            "skill_id": f"s{index:02d}",
            "score": float(50 - index),
        }
        for index in range(50)
    ]


def retrieval_payload(model_tag: str, domain: str) -> JsonObject:
    """Build one complete routed top-50 payload."""

    results: list[JsonObject] = [
        {
            "instance_id": instance_id,
            "gold_skill_ids": ["s00"],
            "retrieved": retrieved_items(),
        }
        for instance_id in instance_ids(domain)
    ]
    return {
        "metadata": {
            "dataset": domain,
            "top_k": 50,
            "n_queries": len(results),
            "retriever": "naive_hyde",
            "router": router_metadata(domain),
            "k_ablation": k_stamp(model_tag, domain, "routed"),
        },
        "metrics": {},
        "results": results,
    }


def gate_fixture(
    root: Path,
    model_tag: str,
    domain: str,
    retrieval: JsonObject,
) -> tuple[Path, Path, Path, dict[str, list[str]]]:
    """Write signals, taus, and gated retrieval for one domain."""

    domain_ids: tuple[str, str] = instance_ids(domain)
    signals_path: Path = root / "gate" / f"{domain}-signals.json"
    taus_path: Path = root / "gate" / f"{domain}-taus.json"
    gated_path: Path = root / "gate" / f"{domain}-gated.json"
    signal_rows: list[JsonObject] = [
        {
            "instance_id": domain_ids[0],
            "top1": "s00",
            "S1": 0.8,
            "S2": 0.5,
            "rel_truth_wrong": False,
        },
        {
            "instance_id": domain_ids[1],
            "top1": "s00",
            "S1": 0.2,
            "S2": 0.5,
            "rel_truth_wrong": False,
        },
    ]
    selected_validation_ids: frozenset[str] = validation_ids(
        frozenset(domain_ids)
    )
    write_json(
        signals_path,
        {
            "signals": signal_rows,
            "cache_misses": 0,
            "encoder": "fixture-encoder",
        },
    )
    write_json(
        taus_path,
        {
            "tau1": 0.5,
            "tau2": None,
            "p_min": 0.9,
            "val_ids": sorted(selected_validation_ids),
            "n_val": len(selected_validation_ids),
        },
    )
    gated: JsonObject = copy.deepcopy(retrieval)
    gated_metadata: JsonObject = cast(JsonObject, gated["metadata"])
    gated_metadata["gate"] = {
        "tau1": 0.5,
        "tau2": None,
        "blocked": 1,
        "skipped": 0,
        "kept": 1,
    }
    gated_results: list[JsonObject] = cast(
        list[JsonObject],
        gated["results"],
    )
    gated_results[1]["retrieved"] = []
    write_json(gated_path, gated)
    return (
        signals_path,
        taus_path,
        gated_path,
        {
            domain_ids[0]: ["s00"],
            domain_ids[1]: [],
        },
    )


def loading_row(
    model_tag: str,
    domain: str,
    instance_id: str,
    arm: str,
    expected: list[str],
    is_validation: bool,
    source_sha256: str,
    failure_category: str,
) -> JsonObject:
    """Build one valid loading decision."""

    hit: bool | None = True if expected else None
    return {
        "schema_version": "k2-loading-decision-v1",
        "instance_id": instance_id,
        "model": model_tag,
        "domain": domain,
        "arm": arm,
        "expected_skill_ids": expected,
        "gold_skill_ids": ["s00"],
        "loaded": bool(expected),
        "hit": hit,
        "gold_loaded": hit is True,
        "is_validation": is_validation,
        "failure_category": failure_category,
        "decision_source_sha256": source_sha256,
    }


def answer_row(
    model_tag: str,
    domain: str,
    instance_id: str,
    arm: str,
    expected: list[str],
    structural_source: JsonObject | None,
) -> JsonObject:
    """Build one answer record."""

    row: JsonObject = {
        "schema_version": "k2-answer-record-v1",
        "instance_id": instance_id,
        "dataset": domain,
        "method": arm,
        "model": model_tag,
        "served_model": "fixture-served-model",
        "raw_output": "fixture answer",
        "skill_ids_used": expected,
        "expected_skill_ids": expected,
        "request_hash": hashlib.sha256(
            f"{model_tag}:{domain}:{arm}:{instance_id}".encode()
        ).hexdigest(),
        "failure_category": "success",
    }
    if structural_source is not None:
        row["provisional_source"] = structural_source
    return row


def evaluation_payload(
    answer_path: Path,
    model_tag: str,
    domain: str,
    arm: str,
    expected_by_id: Mapping[str, list[str]],
) -> JsonObject:
    """Build an evaluation bound to one answer JSONL."""

    selected_validation_ids: frozenset[str] = validation_ids(
        frozenset(instance_ids(domain))
    )
    details: list[JsonObject] = []
    for index, instance_id in enumerate(instance_ids(domain)):
        request_hash: str = hashlib.sha256(
            f"{model_tag}:{domain}:{arm}:{instance_id}".encode()
        ).hexdigest()
        details.append(
            {
                "schema_version": "k2-answer-evaluation-row-v1",
                "instance_id": instance_id,
                "model": model_tag,
                "served_model": "fixture-served-model",
                "domain": domain,
                "arm": arm,
                "correct": index == 0,
                "failure_category": "success",
                "request_hash": request_hash,
                "expected_skill_ids": expected_by_id[instance_id],
                "skill_ids_used": expected_by_id[instance_id],
                "is_validation": instance_id in selected_validation_ids,
                "raw_output_sha256": "4" * 64,
            }
        )
    heldout: list[JsonObject] = [
        row for row in details if row["is_validation"] is False
    ]

    def metrics(rows: Sequence[JsonObject]) -> JsonObject:
        correct: int = sum(row["correct"] is True for row in rows)
        return {
            "total": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "failure_categories": {"success": len(rows)},
        }

    return {
        "schema_version": "k2-answer-evaluation-v1",
        "model": model_tag,
        "served_model": "fixture-served-model",
        "domain": domain,
        "arm": arm,
        "metrics": {
            "full": metrics(details),
            "heldout": metrics(heldout),
        },
        "provenance": {
            "answers_sha256": sha256_path(answer_path),
            "audit_sha256": "5" * 64,
            "instances_sha256": "6" * 64,
            "validation_source_sha256": "7" * 64,
        },
        "details": details,
    }


def write_reuse_audit(
    formal_dir: Path,
    model_tag: str,
    domain: str,
    arm: str,
    expected_by_id: Mapping[str, list[str]],
    answer_path: Path,
) -> None:
    """Write one reuse audit and matching completion proof."""

    file_arm: str = arm.replace("_", "-")
    reuse_path: Path = (
        formal_dir / "audits" / f"{domain}-{file_arm}.reuse.jsonl"
    )
    completion_path: Path = (
        formal_dir / "audits" / f"{domain}-{file_arm}.completion.json"
    )
    rows: list[JsonObject] = [
        {
            "instance_id": instance_id,
            "arm": arm,
            "expected_skill_ids": expected,
            "status": "needs_inference",
            "reason": "legacy_record_missing",
            "new_request_hash": hashlib.sha256(
                f"{model_tag}:{domain}:{arm}:{instance_id}".encode()
            ).hexdigest(),
        }
        for instance_id, expected in expected_by_id.items()
    ]
    write_jsonl(reuse_path, rows)
    write_json(
        completion_path,
        {
            "schema_version": "k2-answer-validation-v1",
            "valid": True,
            "model": CACHE_MODEL_TAGS[model_tag],
            "domain": domain,
            "arm": arm,
            "expected": len(rows),
            "observed": len(rows),
            "reused_same_arm": 0,
            "new_records": len(rows),
            "answers_sha256": sha256_path(answer_path),
            "audit_sha256": sha256_path(reuse_path),
        },
    )


def write_answer_job(
    formal_dir: Path,
    model_tag: str,
    domain: str,
    arm: str,
    expected_by_id: Mapping[str, list[str]],
    structural_sources: Mapping[str, JsonObject],
) -> list[JsonObject]:
    """Write answer, evaluation, and reuse artifacts for one job."""

    file_arm: str = arm.replace("_", "-")
    answer_path: Path = formal_dir / f"{domain}-{file_arm}.jsonl"
    evaluation_path: Path = (
        formal_dir / f"{domain}-{file_arm}.eval.json"
    )
    rows: list[JsonObject] = [
        answer_row(
            model_tag,
            domain,
            instance_id,
            arm,
            expected_by_id[instance_id],
            structural_sources.get(instance_id),
        )
        for instance_id in instance_ids(domain)
    ]
    write_jsonl(answer_path, rows)
    evaluation: JsonObject = evaluation_payload(
        answer_path,
        model_tag,
        domain,
        arm,
        expected_by_id,
    )
    write_json(evaluation_path, evaluation)
    write_reuse_audit(
        formal_dir,
        model_tag,
        domain,
        arm,
        expected_by_id,
        answer_path,
    )
    return cast(list[JsonObject], evaluation["details"])


def write_select_artifacts(
    formal_dir: Path,
    model_tag: str,
    domain: str,
    retrieval_path: Path,
) -> tuple[dict[str, list[str]], list[JsonObject]]:
    """Write selection, selected-source, and loading fixtures."""

    selected_by_id: dict[str, list[str]] = {
        instance_id: ["s00"] for instance_id in instance_ids(domain)
    }
    selection_path: Path = (
        formal_dir / f"{domain}-routed-select.selection.jsonl"
    )
    selected_source_path: Path = (
        formal_dir / f"{domain}-routed-select-source.json"
    )
    selection_rows: list[JsonObject] = [
        {
            "schema_version": "k2-selection-record-v1",
            "instance_id": instance_id,
            "dataset": domain,
            "arm": "routed_select",
            "model": CACHE_MODEL_TAGS[model_tag],
            "ordered_candidate_ids": [
                f"s{index:02d}" for index in range(50)
            ],
            "candidate_hash": "8" * 64,
            "selector_request_hash": hashlib.sha256(
                f"select:{model_tag}:{domain}:{instance_id}".encode()
            ).hexdigest(),
            "selected_skill_id": "s00",
            "selected_rank": 1,
            "rank1_fallback": False,
            "failure_category": "success",
            "source_sha256": sha256_path(retrieval_path),
        }
        for instance_id in instance_ids(domain)
    ]
    write_jsonl(selection_path, selection_rows)
    write_json(
        selected_source_path,
        {
            "metadata": {"dataset": domain},
            "metrics": {},
            "results": [
                {
                    "instance_id": instance_id,
                    "gold_skill_ids": ["s00"],
                    "retrieved": [
                        {"skill_id": "s00", "score": 50.0}
                    ],
                }
                for instance_id in instance_ids(domain)
            ],
        },
    )
    selected_source_sha256: str = sha256_path(selected_source_path)
    attempt_log_path: Path = (
        formal_dir / "logs" / f"{domain}-routed-select.attempts.jsonl"
    )
    attempt_rows: list[JsonObject] = [
        {
            "selector_request_hash": row["selector_request_hash"],
            "attempt": 1,
        }
        for row in selection_rows
    ]
    write_jsonl(attempt_log_path, attempt_rows)
    write_json(
        formal_dir
        / "audits"
        / f"{domain}-routed-select.selection-completion.json",
        {
            "schema_version": "k2-selection-validation-v1",
            "valid": True,
            "model": CACHE_MODEL_TAGS[model_tag],
            "domain": domain,
            "expected": len(selection_rows),
            "observed": len(selection_rows),
            "failure_categories": {"success": len(selection_rows)},
            "attempt_records": len(attempt_rows),
            "selection_sha256": sha256_path(selection_path),
            "selected_source_sha256": selected_source_sha256,
            "attempt_log_sha256": sha256_path(attempt_log_path),
        },
    )
    validation_set: frozenset[str] = validation_ids(
        frozenset(instance_ids(domain))
    )
    loading_rows: list[JsonObject] = [
        loading_row(
            model_tag,
            domain,
            instance_id,
            "routed_select",
            ["s00"],
            instance_id in validation_set,
            selected_source_sha256,
            "success",
        )
        for instance_id in instance_ids(domain)
    ]
    write_jsonl(
        formal_dir / f"{domain}-routed-select.loading.jsonl",
        loading_rows,
    )
    return selected_by_id, loading_rows


def write_fixed_source(
    source_root: Path,
    model_tag: str,
    domain: str,
) -> tuple[dict[str, list[str]], Path]:
    """Write one Qwen4 fixed-gated decision source."""

    expected: dict[str, list[str]] = {
        instance_id: ["s01"] for instance_id in instance_ids(domain)
    }
    source_path: Path = source_root / f"{domain}-fixed-gated.json"
    write_json(
        source_path,
        {
            "metadata": {
                "dataset": domain,
                "top_k": 50,
                "n_queries": 2,
                "k_ablation": k_stamp(
                    model_tag,
                    domain,
                    "naive_skill",
                ),
            },
            "metrics": {},
            "results": [
                {
                    "instance_id": instance_id,
                    "gold_skill_ids": ["s00"],
                    "retrieved": [
                        {"skill_id": "s01", "score": 49.0}
                    ],
                }
                for instance_id in instance_ids(domain)
            ],
        },
    )
    return expected, source_path


def loading_metric_rows(
    model_tag: str,
    rows: Sequence[JsonObject],
) -> list[JsonObject]:
    """Build exact per-model loading long metrics."""

    arms: tuple[str, ...] = (
        ("routed_always", "routed_gated", "routed_select")
        if model_tag in SELECT_ELIGIBLE_MODELS
        else ("routed_always", "routed_gated")
    )
    output: list[JsonObject] = []
    for split in ("full", "heldout"):
        split_rows: list[JsonObject] = (
            list(rows)
            if split == "full"
            else [
                row
                for row in rows
                if row["is_validation"] is False
            ]
        )
        for arm in arms:
            arm_rows: list[JsonObject] = [
                row for row in split_rows if row["arm"] == arm
            ]
            pooled = compute_loading_metrics(cast(Sequence, arm_rows))
            output.append(
                {
                    "schema_version": "k2-loading-metrics-v1",
                    "level": "per_model_pooled",
                    "split": split,
                    "model": model_tag,
                    "domain": None,
                    "arm": arm,
                    "support": None,
                    **pooled,
                }
            )
            for domain in RULE_DOMAINS:
                domain_rows: list[JsonObject] = [
                    row for row in arm_rows if row["domain"] == domain
                ]
                metrics = compute_loading_metrics(cast(Sequence, domain_rows))
                output.append(
                    {
                        "schema_version": "k2-loading-metrics-v1",
                        "level": "per_model_domain",
                        "split": split,
                        "model": model_tag,
                        "domain": domain,
                        "arm": arm,
                        "support": None,
                        **metrics,
                    }
                )
    return output


def answer_metric_rows(
    model_tag: str,
    rows: Sequence[JsonObject],
    arms: Sequence[str],
) -> list[JsonObject]:
    """Build exact per-model answer long metrics."""

    output: list[JsonObject] = []
    for split in ("full", "heldout"):
        split_rows: list[JsonObject] = (
            list(rows)
            if split == "full"
            else [
                row
                for row in rows
                if row["is_validation"] is False
            ]
        )
        for arm in arms:
            arm_rows: list[JsonObject] = [
                row for row in split_rows if row["arm"] == arm
            ]
            for level, domain in (
                ("model_pooled", None),
                *(
                    ("model_domain", rule_domain)
                    for rule_domain in RULE_DOMAINS
                ),
            ):
                selected: list[JsonObject] = [
                    row
                    for row in arm_rows
                    if domain is None or row["domain"] == domain
                ]
                correct: int = sum(
                    row["correct"] is True for row in selected
                )
                output.append(
                    {
                        "schema_version": "k2-answer-metrics-long-v1",
                        "level": level,
                        "split": split,
                        "model": model_tag,
                        "domain": domain,
                        "arm": arm,
                        "support": None,
                        "n": len(selected),
                        "correct": correct,
                        "accuracy": correct / len(selected),
                        "failure_categories": {
                            "success": len(selected)
                        },
                    }
                )
    return output


def answer_summary(
    model_tag: str,
    metric_rows: Sequence[JsonObject],
    arms: Sequence[str],
) -> JsonObject:
    """Build the summarizer's nested per-model block."""

    model_block: JsonObject = {}
    for split in ("full", "heldout"):
        split_block: JsonObject = {}
        for arm in arms:
            split_block[arm] = next(
                row
                for row in metric_rows
                if row["level"] == "model_pooled"
                and row["split"] == split
                and row["arm"] == arm
            )
        model_block[split] = split_block
    return {
        "schema_version": "k2-answer-summary-v1",
        "model_pooled": {model_tag: model_block},
        "fleet_model_macro": {},
    }


def significance_payload(
    model_tag: str,
    arms: Sequence[str],
) -> JsonObject:
    """Build all registered comparisons involving the fixture model."""

    specifications: dict[str, tuple[str, str]] = {
        "gated_vs_always_seven_model": (
            "routed_gated",
            "routed_always",
        )
    }
    if "routed_select" in arms:
        specifications.update(
            {
                "gated_vs_select_five_model": (
                    "routed_gated",
                    "routed_select",
                ),
                "select_vs_always_five_model": (
                    "routed_select",
                    "routed_always",
                ),
            }
        )
    if "fixed_gated" in arms:
        specifications["routed_gated_vs_fixed_gated"] = (
            "routed_gated",
            "fixed_gated",
        )
    comparisons: JsonObject = {
        name: {
            "schema_version": "k2-answer-paired-comparison-v1",
            "split": "heldout",
            "arm_a": arm_a,
            "arm_b": arm_b,
            "models": [model_tag],
            "by_model": {
                model_tag: {
                    "by_domain": {},
                    "pooled_instances": {
                        "difference": 0.0,
                        "ci95": [-0.1, 0.1],
                        "p_two_sided": 1.0,
                    },
                }
            },
        }
        for name, (arm_a, arm_b) in specifications.items()
    }
    return {
        "schema_version": "k2-answer-paired-comparisons-v1",
        "split": "heldout",
        "bootstrap_samples": 32,
        "bootstrap_seed": 0,
        "comparisons": comparisons,
    }


def write_early_raw_source(
    root: Path,
    model_tag: str,
    domain: str,
    arm: str,
) -> tuple[JsonObject, dict[str, JsonObject]]:
    """Write one recovered raw source and its per-line references."""

    provenance_dir: Path = root / "provenance"
    source_path: Path = (
        provenance_dir / "raw-answers" / f"{domain}-{arm}.jsonl"
    )
    source_rows: list[JsonObject] = [
        {
            "model": model_tag,
            "domain": domain,
            "arm": arm,
            "instance_id": instance_id,
            "raw_output": "fixture answer",
        }
        for instance_id in instance_ids(domain)
    ]
    write_jsonl(source_path, source_rows)
    source_sha256: str = sha256_path(source_path)
    references: dict[str, JsonObject] = {}
    for line_number, (instance_id, source_row) in enumerate(
        zip(instance_ids(domain), source_rows, strict=True),
        start=1,
    ):
        source_line: str = canonical_json(source_row)
        references[instance_id] = {
            "source_path": (
                f"/fixture-server/raw-answers/{source_path.name}"
            ),
            "source_sha256": source_sha256,
            "source_line_number": line_number,
            "source_line_sha256": hashlib.sha256(
                source_line.encode("utf-8")
            ).hexdigest(),
        }
    source_entry: JsonObject = {
        "path": source_path.relative_to(provenance_dir).as_posix(),
        "sha256": source_sha256,
        "rows": len(source_rows),
    }
    return source_entry, references


def count_provenance_values(
    rows: Sequence[JsonObject],
    field: str,
) -> dict[str, int]:
    """Count one required string field across provenance rows."""

    counts: dict[str, int] = {}
    for row in rows:
        value: object = row[field]
        if not isinstance(value, str):
            raise TypeError(
                f"Fixture provenance field must be a string: "
                f"field={field}, value={value!r}"
            )
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_answer_provenance(
    root: Path,
    model_tag: str,
    rows: Sequence[JsonObject],
    source_files: Sequence[JsonObject],
) -> Path:
    """Write one explicit fixture answer-provenance manifest."""

    path: Path = root / "provenance" / "answer-provenance.json"
    write_json(
        path,
        {
            "schema_version": "k2-answer-provenance-manifest-v1",
            "model": model_tag,
            "raw_sources_verified": bool(source_files),
            "source_files": list(source_files),
            "rows": list(rows),
            "cohort_counts": count_provenance_values(rows, "cohort"),
            "provenance_level_counts": count_provenance_values(
                rows,
                "provenance_level",
            ),
        },
    )
    return path


def build_fixture(root: Path, model_tag: str) -> FixturePack:
    """Build a complete five-domain, two-instance public-pack fixture."""

    formal_dir: Path = root / "formal"
    (formal_dir / "audits").mkdir(parents=True)
    (formal_dir / "logs").mkdir()
    retrieval_paths: dict[str, Path] = {}
    router_paths: dict[str, Path] = {}
    signals_paths: dict[str, Path] = {}
    taus_paths: dict[str, Path] = {}
    gated_paths: dict[str, Path] = {}
    fixed_gated_paths: dict[str, Path] = {}
    gated_expected: dict[str, dict[str, list[str]]] = {}
    all_loading_rows: list[JsonObject] = []
    all_answer_details: list[JsonObject] = []
    provenance_rows: list[JsonObject] = []
    provenance_source_files: list[JsonObject] = []
    arms: list[str] = ["routed_always", "routed_gated"]
    if model_tag in SELECT_ELIGIBLE_MODELS:
        arms.append("routed_select")
    if model_tag == FIXED_MODEL:
        arms.append("fixed_gated")

    for domain in RETRIEVAL_DOMAINS:
        payload: JsonObject = retrieval_payload(model_tag, domain)
        retrieval_path: Path = (
            root / "retrieval" / f"{domain}-routed.json"
        )
        router_path: Path = root / "router" / f"{domain}-router.json"
        write_json(retrieval_path, payload)
        write_json(router_path, payload)
        retrieval_paths[domain] = retrieval_path
        router_paths[domain] = router_path
        if domain not in RULE_DOMAINS:
            continue
        (
            signals_paths[domain],
            taus_paths[domain],
            gated_paths[domain],
            gated_expected[domain],
        ) = gate_fixture(root, model_tag, domain, payload)
        validation_set: frozenset[str] = validation_ids(
            frozenset(instance_ids(domain))
        )
        deterministic_rows: list[JsonObject] = []
        for instance_id in instance_ids(domain):
            deterministic_rows.append(
                loading_row(
                    model_tag,
                    domain,
                    instance_id,
                    "routed_always",
                    ["s00"],
                    instance_id in validation_set,
                    sha256_path(retrieval_path),
                    "success",
                )
            )
            deterministic_rows.append(
                loading_row(
                    model_tag,
                    domain,
                    instance_id,
                    "routed_gated",
                    gated_expected[domain][instance_id],
                    instance_id in validation_set,
                    sha256_path(gated_paths[domain]),
                    "success",
                )
            )
        write_jsonl(
            formal_dir / f"{domain}-routed-always-gated.loading.jsonl",
            deterministic_rows,
        )
        all_loading_rows.extend(deterministic_rows)
        selected_by_id: dict[str, list[str]] = {}
        if model_tag in SELECT_ELIGIBLE_MODELS:
            selected_by_id, select_loading = write_select_artifacts(
                formal_dir,
                model_tag,
                domain,
                retrieval_path,
            )
            all_loading_rows.extend(select_loading)
        fixed_by_id: dict[str, list[str]] = {}
        if model_tag == FIXED_MODEL:
            fixed_by_id, fixed_gated_paths[domain] = write_fixed_source(
                root / "fixed-gated",
                model_tag,
                domain,
            )
        for arm in arms:
            if arm == "routed_always":
                expected_by_id: Mapping[str, list[str]] = {
                    instance_id: ["s00"]
                    for instance_id in instance_ids(domain)
                }
            elif arm == "routed_gated":
                expected_by_id = gated_expected[domain]
            elif arm == "routed_select":
                expected_by_id = selected_by_id
            else:
                expected_by_id = fixed_by_id
            structural_sources: Mapping[str, JsonObject] = {}
            provenance_level: str = "formal_direct"
            provenance_cohort: str = "formal_k2"
            if (
                model_tag in {"deepseek7b", "llama31-8b", "yi15-9b"}
                and arm in {"routed_always", "routed_gated"}
            ):
                source_file, source_references = write_early_raw_source(
                    root,
                    model_tag,
                    domain,
                    arm,
                )
                provenance_source_files.append(source_file)
                structural_sources = source_references
                provenance_level = "posthoc_structural"
                provenance_cohort = "early_raw_k2"
            provenance_rows.extend(
                {
                    "schema_version": "k2-answer-provenance-row-v1",
                    "model": model_tag,
                    "domain": domain,
                    "arm": arm,
                    "instance_id": instance_id,
                    "provenance_level": provenance_level,
                    "cohort": provenance_cohort,
                }
                for instance_id in instance_ids(domain)
            )
            all_answer_details.extend(
                write_answer_job(
                    formal_dir,
                    model_tag,
                    domain,
                    arm,
                    expected_by_id,
                    structural_sources,
                )
            )

    marker_prefix: str = (
        "K2_ELIGIBLE_MODEL_FORMAL_COMPLETE"
        if model_tag in SELECT_ELIGIBLE_MODELS
        else "K2_INELIGIBLE_MODEL_FORMAL_COMPLETE"
    )
    completion_count: int = len(arms) * len(RULE_DOMAINS)
    if model_tag in SELECT_ELIGIBLE_MODELS:
        completion_count += len(RULE_DOMAINS)
    (formal_dir / "FORMAL_COMPLETE").write_text(
        f"{marker_prefix} result_tag={model_tag} "
        f"completions={completion_count}\n",
        encoding="utf-8",
    )

    loading_metrics_path: Path = root / "fleet-loading.jsonl"
    answer_metrics_path: Path = root / "fleet-answer.jsonl"
    answer_summary_path: Path = root / "fleet-answer-summary.json"
    significance_path: Path = root / "fleet-significance.json"
    loading_rows: list[JsonObject] = loading_metric_rows(
        model_tag,
        all_loading_rows,
    )
    answer_rows: list[JsonObject] = answer_metric_rows(
        model_tag,
        all_answer_details,
        arms,
    )
    write_jsonl(loading_metrics_path, loading_rows)
    write_jsonl(answer_metrics_path, answer_rows)
    write_json(
        answer_summary_path,
        answer_summary(model_tag, answer_rows, arms),
    )
    write_json(
        significance_path,
        significance_payload(model_tag, arms),
    )
    answer_provenance_path: Path = write_answer_provenance(
        root,
        model_tag,
        provenance_rows,
        provenance_source_files,
    )
    return {
        "model_tag": model_tag,
        "formal_dir": formal_dir,
        "retrieval_paths": retrieval_paths,
        "router_paths": router_paths,
        "signals_paths": signals_paths,
        "taus_paths": taus_paths,
        "gated_paths": gated_paths,
        "fixed_gated_paths": fixed_gated_paths,
        "loading_metrics_path": loading_metrics_path,
        "answer_metrics_path": answer_metrics_path,
        "answer_summary_path": answer_summary_path,
        "significance_path": significance_path,
        "answer_provenance_path": answer_provenance_path,
    }


def cli_arguments(fixture: FixturePack, output_dir: Path) -> list[str]:
    """Build one fully explicit exporter command."""

    arguments: list[str] = [
        "scripts/export_k2_public_pack.py",
        "--model-tag",
        fixture["model_tag"],
        "--formal-dir",
        str(fixture["formal_dir"]),
    ]
    for domain in RETRIEVAL_DOMAINS:
        arguments.extend(
            [
                f"--{domain}-retrieval",
                str(fixture["retrieval_paths"][domain]),
                f"--{domain}-router",
                str(fixture["router_paths"][domain]),
            ]
        )
    for domain in RULE_DOMAINS:
        arguments.extend(
            [
                f"--{domain}-signals",
                str(fixture["signals_paths"][domain]),
                f"--{domain}-taus",
                str(fixture["taus_paths"][domain]),
                f"--{domain}-gated",
                str(fixture["gated_paths"][domain]),
            ]
        )
        fixed_gated_path: Path | None = fixture["fixed_gated_paths"].get(
            domain
        )
        if fixed_gated_path is not None:
            arguments.extend(
                [
                    f"--{domain}-fixed-gated",
                    str(fixed_gated_path),
                ]
            )
    arguments.extend(
        [
            "--loading-metrics-long",
            str(fixture["loading_metrics_path"]),
            "--answer-metrics-long",
            str(fixture["answer_metrics_path"]),
            "--answer-summary",
            str(fixture["answer_summary_path"]),
            "--significance",
            str(fixture["significance_path"]),
            "--answer-provenance",
            str(fixture["answer_provenance_path"]),
            "--output-dir",
            str(output_dir),
        ]
    )
    return arguments


def run_cli(
    fixture: FixturePack,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Run the public-pack exporter without hiding failures."""

    environment: dict[str, str] = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    return subprocess.run(
        [sys.executable, *cli_arguments(fixture, output_dir)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def assert_success(completed: subprocess.CompletedProcess[str]) -> None:
    """Assert a CLI invocation succeeded with useful diagnostics."""

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_select_eligible_pack_is_complete_reproducible_and_manifested(
    tmp_path: Path,
) -> None:
    """Export the 12-file eligible pack and verify deterministic gzip."""

    fixture: FixturePack = build_fixture(
        tmp_path / "fixture",
        "llama31-8b",
    )
    first_output: Path = tmp_path / "first" / "k2"
    second_output: Path = tmp_path / "second" / "k2"
    first_output.parent.mkdir()
    second_output.parent.mkdir()
    assert_success(run_cli(fixture, first_output))
    assert_success(run_cli(fixture, second_output))
    expected_names: set[str] = {
        "retrieval_top50.jsonl.gz",
        "router_decisions.json",
        "gating_per_instance.jsonl.gz",
        "loading_per_instance.jsonl.gz",
        "selection_per_instance.jsonl.gz",
        "answer_per_instance.jsonl.gz",
        "answer_metrics.json",
        "metrics_flat.jsonl.gz",
        "significance.json",
        "reuse_manifest.json",
        "manifest.json",
        "README.md",
    }
    assert {path.name for path in first_output.iterdir()} == expected_names
    for filename in sorted(expected_names):
        assert (first_output / filename).read_bytes() == (
            second_output / filename
        ).read_bytes()
    for path in first_output.glob("*.gz"):
        assert path.read_bytes()[:2] == b"\x1f\x8b"
        assert read_gzip_rows(path)
    manifest: JsonObject = json.loads(
        (first_output / "manifest.json").read_text(encoding="utf-8")
    )
    files: JsonObject = cast(JsonObject, manifest["files"])
    assert "manifest.json" not in files
    assert manifest["manifest_self_policy"] == {
        "included_in_files": False,
        "reason": (
            "manifest.json is excluded because a file cannot contain "
            "its own stable SHA-256 digest."
        ),
    }
    formal_completion: JsonObject = cast(
        JsonObject,
        manifest["formal_completion"],
    )
    assert formal_completion["policy"] == (
        "producer_marker_plus_per_job_completion_audits"
    )
    assert cast(JsonObject, formal_completion["marker"])["present"] is True
    for filename, raw_record in files.items():
        record: JsonObject = cast(JsonObject, raw_record)
        assert record["sha256"] == sha256_path(first_output / filename)
        assert isinstance(record["rows"], int)
        assert isinstance(record["schema"], str)
        assert record["provenance_level"] in {
            "per_instance",
            "per_domain",
            "per_model",
            "audit",
            "documentation",
        }


def test_select_ineligible_pack_omits_selection_and_marks_unavailable(
    tmp_path: Path,
) -> None:
    """Export DeepSeek without manufacturing empty Select evidence."""

    fixture: FixturePack = build_fixture(
        tmp_path / "fixture",
        "deepseek7b",
    )
    output_dir: Path = tmp_path / "public" / "k2"
    output_dir.parent.mkdir()
    assert_success(run_cli(fixture, output_dir))
    assert not (output_dir / "selection_per_instance.jsonl.gz").exists()
    manifest: JsonObject = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    arms: JsonObject = cast(JsonObject, manifest["arms"])
    routed_select: JsonObject = cast(
        JsonObject,
        arms["routed_select"],
    )
    assert routed_select["status"] == "unavailable"
    assert routed_select["accounting"] == (
        "excluded from Select denominators; never zero-filled"
    )


def test_qwen_fixed_gated_is_answer_only_not_routed_loading(
    tmp_path: Path,
) -> None:
    """Keep fixed_gated in answers and metrics but outside loading rows."""

    fixture: FixturePack = build_fixture(
        tmp_path / "fixture",
        FIXED_MODEL,
    )
    output_dir: Path = tmp_path / "public" / "k2"
    output_dir.parent.mkdir()
    assert_success(run_cli(fixture, output_dir))
    answer_arms: set[str] = {
        cast(str, row["arm"])
        for row in read_gzip_rows(
            output_dir / "answer_per_instance.jsonl.gz"
        )
    }
    loading_arms: set[str] = {
        cast(str, row["arm"])
        for row in read_gzip_rows(
            output_dir / "loading_per_instance.jsonl.gz"
        )
    }
    metric_arms: set[str] = {
        cast(str, row["arm"])
        for row in read_gzip_rows(output_dir / "metrics_flat.jsonl.gz")
        if row["metric_family"] == "answer"
    }
    assert "fixed_gated" in answer_arms
    assert "fixed_gated" in metric_arms
    assert loading_arms == {
        "routed_always",
        "routed_gated",
        "routed_select",
    }


def test_missing_aggregate_marker_uses_validated_per_job_audits(
    tmp_path: Path,
) -> None:
    """Accept pre-marker formal runs only through complete per-job audits."""

    fixture: FixturePack = build_fixture(
        tmp_path / "fixture",
        "llama31-8b",
    )
    (fixture["formal_dir"] / "FORMAL_COMPLETE").unlink()
    output_dir: Path = tmp_path / "public" / "k2"
    output_dir.parent.mkdir()
    assert_success(run_cli(fixture, output_dir))
    manifest: JsonObject = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    completion: JsonObject = cast(
        JsonObject,
        manifest["formal_completion"],
    )
    assert completion == {
        "schema_version": "k2-public-formal-completion-evidence-v1",
        "policy": "per_job_completion_audits",
        "marker": {
            "present": False,
            "sha256": None,
            "declared_completions": None,
        },
        "validated_answer_completion_audits": 12,
        "validated_selection_completion_audits": 4,
    }


def test_existing_output_and_unknown_formal_root_fail_closed(
    tmp_path: Path,
) -> None:
    """Reject overwrite and any unclassified formal-root entry."""

    fixture: FixturePack = build_fixture(
        tmp_path / "fixture",
        "llama31-8b",
    )
    output_dir: Path = tmp_path / "public" / "k2"
    output_dir.parent.mkdir()
    assert_success(run_cli(fixture, output_dir))
    overwrite: subprocess.CompletedProcess[str] = run_cli(
        fixture,
        output_dir,
    )
    assert overwrite.returncode != 0
    assert "Refusing to overwrite" in overwrite.stderr

    unknown_fixture: FixturePack = build_fixture(
        tmp_path / "unknown-fixture",
        "llama31-8b",
    )
    (unknown_fixture["formal_dir"] / "mystery.bin").write_bytes(b"x")
    unknown_output: Path = tmp_path / "unknown-public" / "k2"
    unknown_output.parent.mkdir()
    unknown: subprocess.CompletedProcess[str] = run_cli(
        unknown_fixture,
        unknown_output,
    )
    assert unknown.returncode != 0
    assert "unknown entries" in unknown.stderr
    assert not unknown_output.exists()

    malformed_fixture: FixturePack = build_fixture(
        tmp_path / "malformed-marker-fixture",
        "llama31-8b",
    )
    (malformed_fixture["formal_dir"] / "FORMAL_COMPLETE").write_text(
        "NOT_A_FORMAL_MARKER\n",
        encoding="utf-8",
    )
    malformed_output: Path = tmp_path / "malformed-marker-public" / "k2"
    malformed_output.parent.mkdir()
    malformed: subprocess.CompletedProcess[str] = run_cli(
        malformed_fixture,
        malformed_output,
    )
    assert malformed.returncode != 0
    assert "Malformed FORMAL_COMPLETE marker" in malformed.stderr
    assert not malformed_output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("missing_source", "does not exist"),
        ("duplicate_id", "Duplicate retrieval instance ID"),
        ("wrong_model", "field=model"),
        ("wrong_k", "field=k_samples"),
    ),
)
def test_source_identity_and_coverage_errors_fail_before_output(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    """Reject missing, duplicate, wrong-model, and non-K2 retrieval inputs."""

    fixture: FixturePack = build_fixture(
        tmp_path / mutation,
        "llama31-8b",
    )
    theorem_path: Path = fixture["retrieval_paths"]["theoremqa"]
    if mutation == "missing_source":
        fixture["retrieval_paths"]["theoremqa"] = (
            theorem_path.parent / "missing.json"
        )
    else:
        payload: JsonObject = json.loads(
            theorem_path.read_text(encoding="utf-8")
        )
        if mutation == "duplicate_id":
            results: list[JsonObject] = cast(
                list[JsonObject],
                payload["results"],
            )
            results[1]["instance_id"] = results[0]["instance_id"]
        else:
            metadata: JsonObject = cast(JsonObject, payload["metadata"])
            stamp: JsonObject = cast(
                JsonObject,
                metadata["k_ablation"],
            )
            if mutation == "wrong_model":
                stamp["model"] = "other-model"
            else:
                stamp["k_samples"] = 4
        write_json(theorem_path, payload)
    output_dir: Path = tmp_path / f"{mutation}-public" / "k2"
    output_dir.parent.mkdir()
    completed: subprocess.CompletedProcess[str] = run_cli(
        fixture,
        output_dir,
    )
    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not output_dir.exists()
