from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from scripts.evaluate_runtime_matched_baselines import (
    ANSWER_SCHEMA_VERSION,
    EVALUATION_ROW_SCHEMA_VERSION,
    EVALUATION_SCHEMA_VERSION,
    BaselineEvaluationError,
    EvaluationRow,
    JsonObject,
    JsonValue,
    evaluation_row,
    index_rows,
)
from scripts.summarize_runtime_matched_baselines import (
    DOMAIN_COUNTS,
    DOMAIN_VALIDATION_COUNTS,
    DOMAINS,
)


REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]
SEVEN_MODELS: tuple[str, ...] = (
    "deepseek7b",
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
    "yi15-9b",
)
FIVE_MODELS: tuple[str, ...] = (
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
)


def instance_fixture() -> JsonObject:
    """Return one evaluator-compatible instance."""

    return {
        "instance_id": "theoremqa_00000",
        "dataset": "theoremqa",
        "question": "What is 40 + 2?",
        "eval_data": {"answer": "42"},
    }


def answer_fixture(
    failure_category: str,
    raw_output: str,
) -> JsonObject:
    """Return one fresh runtime-matched answer row."""

    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "instance_id": "theoremqa_00000",
        "model": "glm4-9b",
        "served_model": "glm4-9b",
        "domain": "theoremqa",
        "arm": "select_bm25",
        "stage": "answer",
        "raw_output": raw_output,
        "expected_skill_ids": ["skill_00"],
        "skill_ids_used": ["skill_00"] if failure_category == "success" else [],
        "answer_payload_hash": "a" * 64,
        "execution_request_hash": "b" * 64,
        "failure_category": failure_category,
        "runtime_manifest_sha256": "c" * 64,
        "reused_same_arm": False,
    }


def test_evaluator_keeps_method_failures_in_denominator() -> None:
    """Resolved method failures are incorrect without calling the evaluator."""

    evaluator_calls: list[str] = []

    def evaluator(raw_output: str, instance: dict[str, object]) -> dict[str, object]:
        evaluator_calls.append(raw_output)
        return {"correct": raw_output.endswith("42.")}

    success: EvaluationRow = evaluation_row(
        "theoremqa_00000",
        instance_fixture(),
        answer_fixture("success", "Therefore, the answer is 42."),
        set(),
        evaluator,
        "glm4-9b",
        "glm4-9b",
        "theoremqa",
        "select_bm25",
    )
    failure: EvaluationRow = evaluation_row(
        "theoremqa_00000",
        instance_fixture(),
        answer_fixture("method_failure", ""),
        {"theoremqa_00000"},
        evaluator,
        "glm4-9b",
        "glm4-9b",
        "theoremqa",
        "select_bm25",
    )
    assert success["correct"] is True
    assert success["is_validation"] is False
    assert failure["correct"] is False
    assert failure["is_validation"] is True
    assert evaluator_calls == ["Therefore, the answer is 42."]


def test_evaluator_rejects_duplicates_reuse_and_unresolved_failures() -> None:
    """Fresh-only evaluation rejects ambiguous or unresolved evidence."""

    instance: JsonObject = instance_fixture()
    with pytest.raises(BaselineEvaluationError, match="duplicate"):
        index_rows([instance, instance], "fixture")

    reused: JsonObject = answer_fixture(
        "success",
        "Therefore, the answer is 42.",
    )
    reused["reused_same_arm"] = True
    with pytest.raises(BaselineEvaluationError, match="must not reuse"):
        evaluation_row(
            "theoremqa_00000",
            instance,
            reused,
            set(),
            lambda output, row: {"correct": True},
            "glm4-9b",
            "glm4-9b",
            "theoremqa",
            "select_bm25",
        )

    unresolved: JsonObject = answer_fixture("infra_transient", "")
    with pytest.raises(BaselineEvaluationError, match="only resolved"):
        evaluation_row(
            "theoremqa_00000",
            instance,
            unresolved,
            set(),
            lambda output, row: {"correct": False},
            "glm4-9b",
            "glm4-9b",
            "theoremqa",
            "select_bm25",
        )


def instance_id(domain: str, index: int) -> str:
    """Return one deterministic protocol-shaped instance ID."""

    return f"{domain}_{index:05d}"


def correctness(arm: str, index: int) -> bool:
    """Return deterministic, non-identical correctness by arm."""

    divisors: dict[str, int] = {
        "routed_gated": 5,
        "routed_select": 4,
        "bare": 2,
        "always_rerank": 3,
        "select_bm25": 7,
    }
    return index % divisors[arm] != 0


def write_k2_public_answers(root: Path) -> list[Path]:
    """Write seven protocol-shaped public K=2 per-instance files."""

    paths: list[Path] = []
    five_model_set: set[str] = set(FIVE_MODELS)
    for model in SEVEN_MODELS:
        path: Path = root / model / "k2" / "answer_per_instance.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as output_file:
            arms: tuple[str, ...] = (
                ("routed_gated", "routed_select")
                if model in five_model_set
                else ("routed_gated",)
            )
            for arm in arms:
                for domain in DOMAINS:
                    validation_count: int = DOMAIN_VALIDATION_COUNTS[domain]
                    for index in range(DOMAIN_COUNTS[domain]):
                        row: JsonObject = {
                            "schema_version": "k2-public-answer-row-v1",
                            "model": model,
                            "served_model": model,
                            "domain": domain,
                            "arm": arm,
                            "instance_id": instance_id(domain, index),
                            "correct": correctness(arm, index),
                            "failure_category": "success",
                            "is_validation": index < validation_count,
                        }
                        output_file.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
        paths.append(path)
    return paths


def baseline_arms(model: str) -> tuple[str, ...]:
    """Return exact fresh baseline support for one model."""

    if model in set(FIVE_MODELS):
        return ("bare", "always_rerank", "select_bm25")
    return ("bare",)


def baseline_detail(
    model: str,
    domain: str,
    arm: str,
    index: int,
) -> JsonObject:
    """Build one fresh per-instance evaluation detail."""

    return {
        "schema_version": EVALUATION_ROW_SCHEMA_VERSION,
        "model": model,
        "served_model": model,
        "domain": domain,
        "arm": arm,
        "instance_id": instance_id(domain, index),
        "correct": correctness(arm, index),
        "failure_category": "success",
        "is_validation": index < DOMAIN_VALIDATION_COUNTS[domain],
        "runtime_manifest_sha256": (
            f"{index % 16:x}" * 64
        ),
    }


def write_baseline_evaluations(root: Path) -> list[Path]:
    """Write all 68 fresh runtime-matched evaluation files."""

    paths: list[Path] = []
    for model in SEVEN_MODELS:
        for arm in baseline_arms(model):
            for domain in DOMAINS:
                details: list[JsonObject] = [
                    baseline_detail(model, domain, arm, index)
                    for index in range(DOMAIN_COUNTS[domain])
                ]
                path: Path = root / model / f"{domain}-{arm}.eval.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "model": model,
                            "domain": domain,
                            "arm": arm,
                            "provenance": {
                                "legacy_compact_baseline_read": False,
                            },
                            "details": details,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
    return paths


def run_cli(
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run one repository CLI with the current interpreter."""

    environment: dict[str, str] = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def summary_arguments(
    k2_paths: Sequence[Path],
    baseline_paths: Sequence[Path],
    output_root: Path,
) -> list[str]:
    """Build one fully explicit summary invocation."""

    return [
        "scripts/summarize_runtime_matched_baselines.py",
        "--k2-answers",
        *[str(path) for path in k2_paths],
        "--baseline-evals",
        *[str(path) for path in baseline_paths],
        "--seven-models",
        ",".join(SEVEN_MODELS),
        "--five-models",
        ",".join(FIVE_MODELS),
        "--expected-k2-files",
        "7",
        "--expected-baseline-eval-files",
        "68",
        "--expected-total-per-model",
        "2830",
        "--expected-heldout-per-model",
        "2265",
        "--bootstrap-samples",
        "16",
        "--bootstrap-seed",
        "0",
        "--output-long",
        str(output_root / "metrics.jsonl"),
        "--output-summary",
        str(output_root / "summary.json"),
        "--output-comparisons",
        str(output_root / "comparisons.json"),
    ]


def test_fresh_summary_uses_k2_split_and_rejects_split_drift(
    tmp_path: Path,
) -> None:
    """Run all four comparisons without reading legacy compact baselines."""

    k2_paths: list[Path] = write_k2_public_answers(tmp_path / "community")
    baseline_paths: list[Path] = write_baseline_evaluations(
        tmp_path / "eval"
    )
    output_root: Path = tmp_path / "output"
    arguments: list[str] = summary_arguments(
        k2_paths,
        baseline_paths,
        output_root,
    )
    completed: subprocess.CompletedProcess[str] = run_cli(arguments)
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    comparisons: JsonObject = cast(
        JsonObject,
        json.loads(
            (output_root / "comparisons.json").read_text(encoding="utf-8")
        ),
    )
    assert comparisons["comparison_count"] == 4
    support: JsonObject = cast(JsonObject, comparisons["support"])
    assert support["fresh_baseline_rows"] == 48110
    provenance: JsonObject = cast(JsonObject, comparisons["provenance"])
    assert provenance["legacy_compact_baseline_read"] is False
    assert provenance["baseline_runtime_identity_gate"] == (
        "fresh_job_bound_manifests"
    )
    reports: JsonObject = cast(JsonObject, comparisons["comparisons"])
    assert set(reports) == {
        "gated_vs_bare_seven_model",
        "gated_vs_native_rerank_five_model",
        "gated_vs_bm25_select_five_model",
        "hyskill_select_vs_bm25_select_five_model",
    }
    first_report: JsonObject = cast(
        JsonObject,
        reports["gated_vs_bare_seven_model"],
    )
    hierarchy: JsonObject = cast(
        JsonObject,
        first_report["fleet_hierarchical"],
    )
    assert hierarchy["resampling_unit"] == (
        "models, then domains, then paired instances"
    )

    drift_path: Path = baseline_paths[0]
    drift_payload: JsonObject = cast(
        JsonObject,
        json.loads(drift_path.read_text(encoding="utf-8")),
    )
    details: list[JsonValue] = cast(list[JsonValue], drift_payload["details"])
    first_detail: JsonObject = cast(JsonObject, details[0])
    first_heldout_detail: JsonObject = cast(
        JsonObject,
        details[DOMAIN_VALIDATION_COUNTS["theoremqa"]],
    )
    first_detail["is_validation"] = False
    first_heldout_detail["is_validation"] = True
    drift_path.write_text(
        json.dumps(drift_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    failed: subprocess.CompletedProcess[str] = run_cli(arguments)
    assert failed.returncode != 0
    assert "split flag differs from K=2 authority" in failed.stderr
