import json
from pathlib import Path
from typing import cast

import pytest

from hyskill.downstream_reuse import (
    DownstreamDataError,
    JsonObject,
    JsonValue,
    sha256_file,
    sha256_text,
)
from hyskill.k2_answer_provenance import (
    PRODUCTION_CONTRACT,
    ProvenanceContract,
    RawSourceData,
    SourceKey,
    build_model_manifest,
    load_qwen_run_logs,
    load_raw_sources,
)


TEST_BUNDLE: str = "a" * 64
TEST_RUNTIME: JsonObject = {
    "model": "yi15-9b",
    "served_model": "yi15-9b",
}


def yi_contract() -> ProvenanceContract:
    """Return a two-instance contract with one direct row and one retry."""

    return ProvenanceContract(
        models=("yi15-9b",),
        domains=("theoremqa",),
        domain_counts={"theoremqa": 2},
        answer_arms={
            "yi15-9b": ("routed_always", "routed_gated"),
        },
        raw_arms={
            "yi15-9b": ("routed_always", "routed_gated"),
        },
        runtime_models={"yi15-9b": "yi15-9b"},
        runtime_identities={"yi15-9b": TEST_RUNTIME},
        answer_code_bundles={"yi15-9b": TEST_BUNDLE},
        early_raw_models=frozenset({"yi15-9b"}),
        qwen_reference_model="qwen3.5-4b-reference",
        yi_model="yi15-9b",
        expected_model_cohorts={
            "yi15-9b": {"early_raw_k2": 4},
        },
        expected_model_levels={
            "yi15-9b": {
                "posthoc_structural": 3,
                "formal_retry_after_import": 1,
            },
        },
        expected_model_provisional_rows={"yi15-9b": 3},
        expected_fleet_cohorts={"early_raw_k2": 4},
        expected_fleet_levels={
            "posthoc_structural": 3,
            "formal_retry_after_import": 1,
        },
        expected_fleet_provisional_rows=3,
    )


def write_jsonl(path: Path, rows: list[JsonObject]) -> list[str]:
    """Write compact JSONL and return the exact serialized lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def raw_rows(arm: str) -> list[JsonObject]:
    """Return two raw answer rows for one fixture arm."""

    return [
        {
            "instance_id": "theoremqa_00000",
            "dataset": "theoremqa",
            "method": arm,
            "model": "yi15-9b",
            "raw_output": "answer",
            "skill_ids_used": ["skill_0"],
        },
        {
            "instance_id": "theoremqa_00001",
            "dataset": "theoremqa",
            "method": arm,
            "model": "yi15-9b",
            "raw_output": "" if arm == "routed_always" else "bare answer",
            "skill_ids_used": ["skill_1"],
        },
    ]


def final_success_row(
    arm: str,
    instance_id: str,
    raw_output: str,
    skill_ids: list[str],
    provisional_source: JsonObject,
) -> JsonObject:
    """Return one structurally imported final answer row."""

    return {
        "schema_version": "k2-answer-record-v1",
        "instance_id": instance_id,
        "dataset": "theoremqa",
        "method": arm,
        "model": "yi15-9b",
        "served_model": "yi15-9b",
        "raw_output": raw_output,
        "skill_ids_used": skill_ids,
        "expected_skill_ids": skill_ids,
        "failure_category": "success",
        "engine_attempts": 1,
        "actual_injection_state": {
            "state": "confirmed_by_engine",
            "skill_ids": skill_ids,
        },
        "runtime_identity": TEST_RUNTIME,
        "answer_code_bundle_sha256": TEST_BUNDLE,
        "provisional_source": provisional_source,
    }


def final_retry_row() -> JsonObject:
    """Return one bounded Yi retry that stayed EmptyModelOutput."""

    return {
        "schema_version": "k2-answer-record-v1",
        "instance_id": "theoremqa_00001",
        "dataset": "theoremqa",
        "method": "routed_always",
        "model": "yi15-9b",
        "served_model": "yi15-9b",
        "raw_output": "",
        "skill_ids_used": [],
        "expected_skill_ids": ["skill_1"],
        "failure_category": "method_failure",
        "engine_attempts": 3,
        "actual_injection_state": {
            "state": "request_submitted",
            "skill_ids": ["skill_1"],
        },
        "runtime_identity": TEST_RUNTIME,
        "answer_code_bundle_sha256": TEST_BUNDLE,
        "error": {
            "exception_name": "EmptyModelOutput",
            "message": "empty after bounded attempts",
            "response_body": "",
            "status_code": None,
        },
    }


def source_reference(
    source_path: Path,
    source_sha256: str,
    line_number: int,
    line: str,
) -> JsonObject:
    """Return a server-shaped immutable raw source reference."""

    return {
        "source_path": (
            "/server/results/k2-main/yi15-9b/raw-answers/"
            f"{source_path.name}"
        ),
        "source_sha256": source_sha256,
        "source_line_number": line_number,
        "source_line_sha256": sha256_text(line),
    }


def build_yi_fixture(
    root: Path,
) -> tuple[Path, list[Path], ProvenanceContract]:
    """Write a complete small raw-to-final Yi fixture."""

    contract: ProvenanceContract = yi_contract()
    formal_root: Path = root / "formal"
    raw_paths: list[Path] = []
    for arm in ("routed_always", "routed_gated"):
        raw_path: Path = (
            root
            / "results"
            / "k2-main"
            / "yi15-9b"
            / "raw-answers"
            / f"theoremqa-{arm}.jsonl"
        )
        rows: list[JsonObject] = raw_rows(arm)
        lines: list[str] = write_jsonl(raw_path, rows)
        source_sha256: str = sha256_file(raw_path)
        raw_paths.append(raw_path)
        final_rows: list[JsonObject] = [
            final_success_row(
                arm,
                "theoremqa_00000",
                "answer",
                ["skill_0"],
                source_reference(
                    raw_path,
                    source_sha256,
                    1,
                    lines[0],
                ),
            )
        ]
        if arm == "routed_always":
            final_rows.append(final_retry_row())
        else:
            final_rows.append(
                final_success_row(
                    arm,
                    "theoremqa_00001",
                    "bare answer",
                    ["skill_1"],
                    source_reference(
                        raw_path,
                        source_sha256,
                        2,
                        lines[1],
                    ),
                )
            )
        write_jsonl(
            formal_root
            / "yi15-9b"
            / f"theoremqa-{arm.replace('_', '-')}.jsonl",
            final_rows,
        )
    return formal_root, raw_paths, contract


def test_builder_verifies_structural_rows_and_formal_retry(
    tmp_path: Path,
) -> None:
    formal_root, raw_paths, contract = build_yi_fixture(tmp_path)
    sources = load_raw_sources(raw_paths, tmp_path, contract)

    manifest: JsonObject = build_model_manifest(
        "yi15-9b",
        formal_root,
        tmp_path,
        sources,
        {},
        contract,
    )

    assert manifest["cohort_counts"] == {"early_raw_k2": 4}
    assert manifest["provenance_level_counts"] == {
        "posthoc_structural": 3,
        "formal_retry_after_import": 1,
    }
    assert manifest["provisional_source_rows"] == 3
    rows: list[JsonObject] = cast(list[JsonObject], manifest["rows"])
    retry_rows: list[JsonObject] = [
        row
        for row in rows
        if row["provenance_level"] == "formal_retry_after_import"
    ]
    assert retry_rows == [
        {
            **retry_rows[0],
            "skill_identity_binding": (
                "raw.skill_ids_used=final.expected_skill_ids="
                "final.actual_injection_state.skill_ids"
            ),
        }
    ]


def test_builder_fails_closed_on_raw_output_change(
    tmp_path: Path,
) -> None:
    formal_root, raw_paths, contract = build_yi_fixture(tmp_path)
    answer_path: Path = (
        formal_root
        / "yi15-9b"
        / "theoremqa-routed-gated.jsonl"
    )
    final_rows: list[JsonObject] = [
        cast(JsonObject, json.loads(line))
        for line in answer_path.read_text(encoding="utf-8").splitlines()
    ]
    final_rows[0]["raw_output"] = "changed"
    write_jsonl(answer_path, final_rows)
    sources = load_raw_sources(raw_paths, tmp_path, contract)

    with pytest.raises(
        DownstreamDataError,
        match="Raw-to-final field mismatch",
    ):
        build_model_manifest(
            "yi15-9b",
            formal_root,
            tmp_path,
            sources,
            {},
            contract,
        )


def qwen_contract() -> ProvenanceContract:
    """Return a one-row Qwen run-history fixture contract."""

    runtime: JsonObject = {
        "model": "qwen3.5-4b-reference",
        "served_model": "qwen3.5-4b",
    }
    return ProvenanceContract(
        models=("qwen3.5-4b-reference",),
        domains=("theoremqa",),
        domain_counts={"theoremqa": 1},
        answer_arms={
            "qwen3.5-4b-reference": ("routed_always",),
        },
        raw_arms={
            "qwen3.5-4b-reference": ("routed_always",),
        },
        runtime_models={"qwen3.5-4b-reference": "qwen3.5-4b"},
        runtime_identities={"qwen3.5-4b-reference": runtime},
        answer_code_bundles={
            "qwen3.5-4b-reference": TEST_BUNDLE,
        },
        early_raw_models=frozenset(),
        qwen_reference_model="qwen3.5-4b-reference",
        yi_model="yi15-9b",
        expected_model_cohorts={
            "qwen3.5-4b-reference": {"formal_k2": 1},
        },
        expected_model_levels={
            "qwen3.5-4b-reference": {"formal_direct": 1},
        },
        expected_model_provisional_rows={
            "qwen3.5-4b-reference": 1,
        },
        expected_fleet_cohorts={"formal_k2": 1},
        expected_fleet_levels={"formal_direct": 1},
        expected_fleet_provisional_rows=1,
    )


def qwen_log_text(engine_line: str) -> str:
    """Return one complete one-row Qwen run-history log."""

    model: str = "qwen3.5-4b-reference"
    raw_path: str = (
        f"results/k2-main/{model}/raw-answers/"
        "theoremqa-routed_always.jsonl"
    )
    return "\n".join(
        [
            (
                "Provider: topk {'source': "
                f"'results/k-ablation/{model}/routed/k2/"
                "theoremqa-routed.json', 'k': 1}"
            ),
            engine_line,
            "Label:    routed_always",
            "Model:    qwen3.5-4b",
            f"  wrote 1 records → {raw_path}",
            "  theoremqa × qwen3.5-4b × routed_always — 1 records",
            f"  Saved: {raw_path.removesuffix('.jsonl')}.eval.json",
            "",
        ]
    )


def test_qwen_run_history_fails_closed_on_generation_mismatch(
    tmp_path: Path,
) -> None:
    contract: ProvenanceContract = qwen_contract()
    raw_path: Path = (
        tmp_path
        / "results"
        / "k2-main"
        / "qwen3.5-4b-reference"
        / "raw-answers"
        / "theoremqa-routed_always.jsonl"
    )
    write_jsonl(
        raw_path,
        [
            {
                "instance_id": "theoremqa_00000",
                "dataset": "theoremqa",
                "method": "routed_always",
                "model": "qwen3.5-4b",
                "raw_output": "answer",
                "skill_ids_used": ["skill_0"],
            }
        ],
    )
    sources: dict[SourceKey, RawSourceData] = load_raw_sources(
        [raw_path],
        tmp_path,
        contract,
    )
    log_path: Path = raw_path.parent / "logs" / "theoremqa-routed_always.log"
    log_path.parent.mkdir()
    log_path.write_text(
        qwen_log_text(
            "Engine:   direct   {'temperature': 0.0, 'max_tokens': 2048}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        DownstreamDataError,
        match="Qwen run-history line mismatch",
    ):
        load_qwen_run_logs(
            [log_path],
            tmp_path,
            sources,
            contract,
        )


def test_production_contract_freezes_global_provenance_counts() -> None:
    assert PRODUCTION_CONTRACT.expected_fleet_cohorts == {
        "early_raw_k2": 16980,
        "formal_k2": 39620,
    }
    assert PRODUCTION_CONTRACT.expected_fleet_levels == {
        "formal_direct": 39620,
        "posthoc_structural": 16873,
        "formal_retry_after_import": 107,
    }
    assert PRODUCTION_CONTRACT.expected_fleet_provisional_rows == 25363
