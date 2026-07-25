import gzip
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from hyskill.downstream_reuse import JsonLike, JsonObject
from scripts.summarize_k2_answer_evaluations import (
    DOMAIN_COUNTS,
    DOMAIN_VALIDATION_COUNTS,
    DOMAINS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
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
FIXED_MODEL: str = "qwen3.5-4b-reference"


def write_json(path: Path, value: JsonLike) -> None:
    """Write one compact fixture JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def instance_id(domain: str, index: int) -> str:
    """Return one deterministic protocol-shaped instance ID."""

    return f"{domain}_{index:05d}"


def correctness(arm: str, index: int) -> bool:
    """Return deterministic, non-identical correctness by arm."""

    divisors: dict[str, int] = {
        "routed_always": 2,
        "routed_gated": 5,
        "routed_select": 4,
        "fixed_gated": 3,
    }
    divisor: int = divisors[arm]
    return index % divisor != 0


def expected_arms(model: str) -> tuple[str, ...]:
    """Return the full active-arm fixture support for one model."""

    arms: list[str] = ["routed_always", "routed_gated"]
    if model in set(FIVE_MODELS):
        arms.append("routed_select")
    if model == FIXED_MODEL:
        arms.append("fixed_gated")
    return tuple(arms)


def write_k2_evaluations(root: Path) -> list[Path]:
    """Write all 80 protocol-shaped K=2 evaluation files."""

    paths: list[Path] = []
    for model in SEVEN_MODELS:
        for domain in DOMAINS:
            count: int = DOMAIN_COUNTS[domain]
            validation_count: int = DOMAIN_VALIDATION_COUNTS[domain]
            for arm in expected_arms(model):
                details: list[JsonObject] = [
                    {
                        "model": model,
                        "domain": domain,
                        "arm": arm,
                        "instance_id": instance_id(domain, index),
                        "correct": correctness(arm, index),
                        "failure_category": "success",
                        "is_validation": index < validation_count,
                    }
                    for index in range(count)
                ]
                path: Path = root / model / f"{domain}-{arm}.eval.json"
                write_json(
                    path,
                    {
                        "schema_version": "k2-answer-evaluation-v1",
                        "details": details,
                    },
                )
                paths.append(path)
    return paths


def compact_row(
    model: str,
    domain: str,
    index: int,
) -> JsonObject:
    """Build one compact baseline row for a model."""

    validation_field: str = (
        "in_calibration_split_routed"
        if model == FIXED_MODEL
        else "in_calibration_split"
    )
    row: JsonObject = {
        "instance_id": instance_id(domain, index),
        "domain": domain,
        "correct_bare": index % 2 == 0,
        validation_field: index < DOMAIN_VALIDATION_COUNTS[domain],
    }
    if model in set(FIVE_MODELS) and model != FIXED_MODEL:
        row["correct_always_rerank"] = index % 3 != 0
        row["correct_select_bm25"] = index % 7 != 0
    return row


def write_compact_baselines(root: Path) -> None:
    """Write seven compact baseline packs and Qwen4 native files."""

    for model in SEVEN_MODELS:
        model_root: Path = root / model / "k4"
        model_root.mkdir(parents=True, exist_ok=True)
        compact_path: Path = model_root / "gating_per_instance.jsonl.gz"
        with gzip.open(compact_path, "wt", encoding="utf-8") as output_file:
            for domain in DOMAINS:
                for index in range(DOMAIN_COUNTS[domain]):
                    output_file.write(
                        json.dumps(
                            compact_row(model, domain, index),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    native_root: Path = root / FIXED_MODEL / "baselines-native"
    native_root.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        count: int = DOMAIN_COUNTS[domain]
        for arm, divisor in (
            ("always_rerank", 3),
            ("select_bm25", 7),
        ):
            details: list[JsonObject] = [
                {
                    "instance_id": instance_id(domain, index),
                    "correct": index % divisor != 0,
                }
                for index in range(count)
            ]
            write_json(
                native_root / f"{domain}-{arm}.eval.json",
                {
                    "dataset": domain,
                    "method": arm,
                    "model": "qwen35-4b",
                    "metrics": {
                        "total": count,
                        "correct": sum(
                            cast(bool, row["correct"]) for row in details
                        ),
                        "accuracy": (
                            sum(cast(bool, row["correct"]) for row in details)
                            / count
                        ),
                    },
                    "details": details,
                },
            )


def run_cli(
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run one repository CLI with the current test interpreter."""

    environment: dict[str, str] = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def cli_arguments(
    evaluation_paths: Sequence[Path],
    community_root: Path,
    output_path: Path,
) -> list[str]:
    """Build one fully explicit baseline-comparison invocation."""

    return [
        "scripts/summarize_k2_baseline_comparisons.py",
        "--k2-evals",
        *[str(path) for path in evaluation_paths],
        "--community-root",
        str(community_root),
        "--seven-models",
        ",".join(SEVEN_MODELS),
        "--five-models",
        ",".join(FIVE_MODELS),
        "--fixed-model",
        FIXED_MODEL,
        "--expected-eval-files",
        "80",
        "--expected-total-per-model",
        "2830",
        "--expected-heldout-per-model",
        "2265",
        "--bootstrap-samples",
        "32",
        "--bootstrap-seed",
        "0",
        "--output",
        str(output_path),
    ]


def test_baseline_comparison_cli_adapts_qwen_and_rejects_id_drift(
    tmp_path: Path,
) -> None:
    """Run all four comparisons and exercise the strict Qwen4 ID gate."""

    evaluation_paths: list[Path] = write_k2_evaluations(
        tmp_path / "k2-evals"
    )
    community_root: Path = tmp_path / "community-results"
    write_compact_baselines(community_root)
    output_path: Path = tmp_path / "baseline-comparisons.json"
    arguments: list[str] = cli_arguments(
        evaluation_paths,
        community_root,
        output_path,
    )

    completed: subprocess.CompletedProcess[str] = run_cli(arguments)
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    payload: JsonObject = cast(
        JsonObject,
        json.loads(output_path.read_text(encoding="utf-8")),
    )
    assert payload["schema_version"] == "k2-baseline-paired-comparisons-v1"
    assert payload["comparison_count"] == 4
    comparisons: JsonObject = cast(JsonObject, payload["comparisons"])
    assert set(comparisons) == {
        "gated_vs_bare_seven_model",
        "gated_vs_native_rerank_five_model",
        "gated_vs_native_select_five_model",
        "hyskill_select_vs_bm25_select_five_model",
    }
    provenance: JsonObject = cast(JsonObject, payload["provenance"])
    baseline_sources: JsonObject = cast(
        JsonObject,
        provenance["baseline_sources"],
    )
    qwen_source: JsonObject = cast(
        JsonObject,
        baseline_sources[FIXED_MODEL],
    )
    assert qwen_source["adapter"] == (
        "qwen4_compact_bare_plus_native_eval_files"
    )
    assert len(cast(list[JsonLike], qwen_source["native_eval_files"])) == 8
    assert provenance["baseline_runtime_identity_gate"] == (
        "not_proven_by_this_script"
    )

    drift_path: Path = (
        community_root
        / FIXED_MODEL
        / "baselines-native"
        / "champ-select_bm25.eval.json"
    )
    drift_payload: JsonObject = cast(
        JsonObject,
        json.loads(drift_path.read_text(encoding="utf-8")),
    )
    drift_details: list[JsonLike] = cast(
        list[JsonLike],
        drift_payload["details"],
    )
    drift_payload["details"] = drift_details[:-1]
    write_json(drift_path, drift_payload)
    failed: subprocess.CompletedProcess[str] = run_cli(arguments)
    assert failed.returncode != 0
    assert "Qwen4 native detail denominator mismatch" in failed.stderr
