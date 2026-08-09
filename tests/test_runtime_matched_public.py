"""Focused tests for path-free runtime-matched public evidence."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import cast

import pytest

from hyskill.runtime_matched_public import (
    JsonObject,
    JsonValue,
    RuntimeMatchedPublicError,
    deterministic_jsonl_gzip,
    public_answer_row,
    public_decision_row,
    public_runtime_job,
    public_usage_job,
)
from scripts.export_runtime_matched_baselines import (
    RuntimeMatchedExportError,
    scan_sensitive_file,
)


def digest(character: str) -> str:
    """Return one test SHA-256 value."""

    return character * 64


def candidates() -> list[str]:
    """Return 50 deterministic candidate IDs."""

    return [f"skill_{index:02d}" for index in range(50)]


def answer_evaluation() -> JsonObject:
    """Return one private evaluation row."""

    return {
        "schema_version": "runtime-matched-baseline-evaluation-row-v1",
        "instance_id": "theoremqa_00000",
        "model": "glm4-9b",
        "served_model": "glm4-9b",
        "domain": "theoremqa",
        "arm": "bare",
        "correct": True,
        "failure_category": "success",
        "is_validation": False,
        "ground_truth": "private answer",
        "raw_output_sha256": digest("a"),
        "answer_payload_hash": digest("b"),
        "execution_request_hash": digest("c"),
        "runtime_manifest_sha256": digest("d"),
        "skill_ids_used": [],
        "evaluator": {"extracted_answer": "private"},
    }


def test_public_answer_removes_ground_truth_and_evaluator() -> None:
    """Expose only reproducibility fields, not answer content."""

    public: JsonObject = public_answer_row(
        answer_evaluation(),
        "glm4-9b",
        "theoremqa",
        "bare",
    )
    assert public["correct"] is True
    assert public["skill_ids_used"] == []
    assert "ground_truth" not in public
    assert "evaluator" not in public
    with pytest.raises(RuntimeMatchedPublicError, match="Bare answer"):
        row: JsonObject = answer_evaluation()
        row["skill_ids_used"] = ["skill_00"]
        public_answer_row(row, "glm4-9b", "theoremqa", "bare")


def decision_row(arm: str) -> JsonObject:
    """Return one private native decision."""

    ordered: list[str] = candidates()
    common: JsonObject = {
        "instance_id": "theoremqa_00000",
        "model": "glm4-9b",
        "served_model": "glm4-9b",
        "domain": "theoremqa",
        "arm": arm,
        "stage": "decision",
        "ordered_candidate_ids": ordered,
        "selected_skill_id": ordered[2],
        "failure_category": "success",
        "candidate_hash": digest("e"),
        "execution_request_hash": digest("f"),
        "runtime_manifest_sha256": digest("1"),
        "raw_response": "private",
        "raw_responses": ["private"],
    }
    if arm == "select_bm25":
        return {
            **common,
            "schema_version": "runtime-matched-select-decision-v1",
            "candidate_source_sha256": digest("2"),
            "selector_payload_hash": digest("3"),
            "selected_rank": 3,
            "rank1_fallback": False,
            "parse_attempts": 1,
        }
    reranked: list[str] = [ordered[2], ordered[0], ordered[1], *ordered[3:]]
    return {
        **common,
        "schema_version": "runtime-matched-rerank-decision-v1",
        "source_sha256": digest("2"),
        "decision_payload_hash": digest("3"),
        "reranked_candidate_ids": reranked,
        "parse_attempts": 1,
    }


@pytest.mark.parametrize("arm", ("select_bm25", "always_rerank"))
def test_public_decision_keeps_candidates_but_removes_raw_text(
    arm: str,
) -> None:
    """Keep the factorized loading evidence without LLM response text."""

    public: JsonObject = public_decision_row(
        decision_row(arm),
        "glm4-9b",
        "theoremqa",
        arm,
    )
    assert public["selected_skill_id"] == "skill_02"
    assert public["selected_original_rank"] == 3
    assert len(cast(list[JsonValue], public["ordered_candidate_ids"])) == 50
    assert "raw_response" not in public
    assert "raw_responses" not in public


def private_manifest() -> JsonObject:
    """Return one job manifest containing fields that must remain private."""

    return {
        "schema_version": "runtime-matched-job-manifest-v1",
        "runtime_facts": {
            "schema_version": "runtime-matched-runtime-facts-v1",
            "job": {
                "job_id": "glm4-9b-theoremqa-bare-v1",
                "result_tag": "glm4-9b",
                "model": "glm4-9b",
                "domain": "theoremqa",
                "arm": "bare",
            },
            "checkpoint": {
                "repository": "ZhipuAI/glm-4-9b-chat",
                "revision": "snapshots/master",
                "path": "/root/private/checkpoint",
                "provenance": "China-hosted mirror",
                "files_manifest_sha256": digest("4"),
            },
            "tokenizer": {
                "artifacts": {"tokenizer.json": digest("5")},
                "chat_template_sha256": digest("6"),
            },
            "endpoint": {
                "api_base": "http://127.0.0.1:8000/v1",
                "served_model": "glm4-9b",
                "process_command": "/root/private/python -m vllm",
                "vllm_version": "0.19.1",
                "dtype": "bfloat16",
                "quantization": "none",
                "max_model_len": 8192,
                "tensor_parallel_size": 1,
                "models_readback": {"data": [{"id": "glm4-9b"}]},
            },
            "software": {
                "python_version": "3.10.12",
                "pytorch_version": "2.10.0+cu128",
                "transformers_version": "5.3.1",
                "cuda_version": "12.8",
                "driver_version": "550.127.05",
            },
            "hardware": {
                "gpu_model": "NVIDIA A100-SXM4-80GB",
                "gpu_uuid": "GPU-private",
            },
            "source": {"sr_agents_revision": "277fd8d"},
        },
        "generation": {
            "temperature": 0.7,
            "max_tokens": 2048,
            "thinking": False,
            "extra_body": None,
        },
        "artifacts": [
            {
                "name": "instances",
                "path": "/root/private/instances.json",
                "size_bytes": 123,
                "sha256": digest("7"),
            }
        ],
        "code_files": [
            {
                "path": "scripts/run_runtime_matched_bare.py",
                "size_bytes": 456,
                "sha256": digest("8"),
            }
        ],
        "code_bundle_sha256": digest("9"),
    }


def test_public_runtime_job_removes_paths_endpoint_and_gpu_uuid() -> None:
    """Publish identity without disclosing the machine layout."""

    public: JsonObject = public_runtime_job(private_manifest(), digest("a"))
    encoded: str = json.dumps(public, sort_keys=True)
    assert "/root/" not in encoded
    assert "127.0.0.1" not in encoded
    assert "GPU-private" not in encoded
    assert public["manifest_sha256"] == digest("a")
    assert cast(JsonObject, public["checkpoint"])["revision"] == (
        "snapshots/master"
    )


def test_public_usage_job_removes_log_path() -> None:
    """Keep actual reported tokens and source SHA, not server paths."""

    private: JsonObject = {
        "model": "glm4-9b",
        "domain": "theoremqa",
        "arm": "bare",
        "stage": "answer",
        "path": "/root/private/attempts.jsonl",
        "sha256": digest("b"),
        "http_calls": 2,
        "response_calls": 1,
        "error_calls": 1,
        "unique_instances": 1,
        "usage_reported_calls": 1,
        "usage_missing_calls": 1,
        "usage_missing_reasons": {"context_error": 1},
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    public: JsonObject = public_usage_job(private)
    assert "path" not in public
    assert public["total_tokens"] == 120
    assert public["source_log_sha256"] == digest("b")


def test_deterministic_jsonl_gzip_has_fixed_bytes() -> None:
    """Canonical rows produce reproducible compressed bytes."""

    rows: list[JsonObject] = [{"b": 2, "a": 1}, {"value": "x"}]
    first: bytes = deterministic_jsonl_gzip(rows)
    second: bytes = deterministic_jsonl_gzip(rows)
    assert first == second
    decoded: list[JsonObject] = [
        cast(JsonObject, json.loads(line))
        for line in gzip.decompress(first).decode("utf-8").splitlines()
    ]
    assert decoded == [{"a": 1, "b": 2}, {"value": "x"}]


@pytest.mark.parametrize(
    "private_field",
    (
        "ground_truth",
        "evaluator",
        "raw_output",
        "raw_response",
        "process_command",
    ),
)
def test_public_pack_scan_rejects_private_fields(
    tmp_path: Path,
    private_field: str,
) -> None:
    """Reject a generated pack if a private field survives projection."""

    path: Path = tmp_path / "public.json"
    path.write_text(
        json.dumps({private_field: "private"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeMatchedExportError,
        match="sensitive tokens",
    ):
        scan_sensitive_file(path)
