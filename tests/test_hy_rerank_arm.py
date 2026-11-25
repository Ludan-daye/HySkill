"""Unit tests for the E7 hy_rerank arm over routed K=2 candidates."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import cast

from hyskill.runtime_matched_execution import sha256_file
from hyskill.runtime_matched_public import public_answer_row, public_decision_row
from hyskill.runtime_matched_rerank import (
    HY_RERANK_ARM,
    RERANK_ARM,
    JsonObject,
    build_rerank_decision,
    build_rerank_inputs,
    load_retrieval_records,
    rerank_generation,
)
from scripts.export_routed_candidates import export_routed_candidates
from tests.test_runtime_matched_rerank import _manifest, _runtime


def _candidates_corpus(count: int = 50) -> tuple[list[JsonObject], dict[str, JsonObject]]:
    items: list[JsonObject] = [
        {
            "skill_id": f"skill_{idx:03d}",
            "name": f"Skill Name {idx}",
            "description": f"Skill description {idx}",
            "content": f"Detailed procedure {idx}",
        }
        for idx in range(1, count + 1)
    ]
    return items, {cast(str, item["skill_id"]): item for item in items}


def test_routed_export_preserves_order(tmp_path: Path) -> None:
    # Deliberately non-lexicographic order and tied/arbitrary scores to prove exact order preservation
    ordered_ids = [f"skill_{(37 * idx) % 50 + 1:03d}" for idx in range(50)]
    assert len(set(ordered_ids)) == 50

    source_gz = tmp_path / "retrieval_top50.jsonl.gz"
    routed_row = {
        "schema_version": "k2-public-retrieval-row-v1",
        "model": "glm4-9b",
        "domain": "theoremqa",
        "instance_id": "theoremqa_00000",
        "variant": "routed",
        "k_samples": 2,
        "gold_skill_ids": ["skill_001"],
        "source_sha256": "a" * 64,
        "retrieved": [
            {"skill_id": sid, "score": float(100 - rank)}
            for rank, sid in enumerate(ordered_ids)
        ],
    }
    other_domain_row = {
        **routed_row,
        "domain": "champ",
        "instance_id": "champ_00000",
    }
    with gzip.open(source_gz, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(routed_row) + "\n")
        handle.write(json.dumps(other_domain_row) + "\n")

    out_json = tmp_path / "exported_theoremqa.json"
    summary = export_routed_candidates(
        source_gz,
        out_json,
        result_tag="glm4-9b",
        domain="theoremqa",
    )
    assert summary["instance_count"] == 1
    assert summary["source_sha256"] == sha256_file(source_gz)
    assert summary["output_sha256"] == sha256_file(out_json)

    manifest_path = Path(str(out_json) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == sha256_file(source_gz)
    assert manifest["output_sha256"] == sha256_file(out_json)

    _payload, records = load_retrieval_records(out_json)
    assert len(records) == 1
    exported_ids = [
        cast(dict[str, object], entry)["skill_id"]
        for entry in cast(list[object], records[0]["retrieved"])
    ]
    assert exported_ids == ordered_ids


def test_hy_rerank_identity_differs_from_always_rerank() -> None:
    items, corpus = _candidates_corpus(60)
    bm25_ids = [cast(str, item["skill_id"]) for item in items[:50]]
    hy_ids = [cast(str, item["skill_id"]) for item in items[10:60]]

    instance: JsonObject = {
        "instance_id": "theoremqa_00000",
        "dataset": "theoremqa",
        "question": "Compute the eigenvalue of the 2x2 matrix.",
    }
    bm25_source: JsonObject = {
        "instance_id": "theoremqa_00000",
        "gold_skill_ids": ["skill_011"],
        "retrieved": [{"skill_id": sid, "score": float(50 - i)} for i, sid in enumerate(bm25_ids)],
    }
    hy_source: JsonObject = {
        "instance_id": "theoremqa_00000",
        "gold_skill_ids": ["skill_011"],
        "retrieved": [{"skill_id": sid, "score": float(50 - i)} for i, sid in enumerate(hy_ids)],
    }
    manifest = _manifest(rerank_generation(None))

    always_inputs = build_rerank_inputs(
        [instance],
        {"theoremqa_00000": bm25_source},
        corpus,
        _runtime([]),
        manifest,
        "e" * 64,
        arm=RERANK_ARM,
    )
    hy_inputs = build_rerank_inputs(
        [instance],
        {"theoremqa_00000": hy_source},
        corpus,
        _runtime([]),
        manifest,
        "e" * 64,
        arm=HY_RERANK_ARM,
    )
    assert always_inputs[0]["candidate_hash"] != hy_inputs[0]["candidate_hash"]
    assert always_inputs[0]["decision_payload_hash"] != hy_inputs[0]["decision_payload_hash"]
    assert always_inputs[0]["execution_request_hash"] != hy_inputs[0]["execution_request_hash"]

    routed_source_sha = "f" * 64
    decision = build_rerank_decision(
        hy_inputs[0],
        {
            "reranked_candidate_ids": list(hy_ids),
            "raw_responses": ["1, 2, 3"],
            "parse_attempts": 1,
            "client_call_attempts": 1,
            "parse_sufficient": False,
            "omitted_candidate_count": 47,
            "failure_category": "success",
            "error": None,
        },
        "glm4-9b",
        "glm4-9b",
        "theoremqa",
        "e" * 64,
        "c" * 64,
        routed_source_sha,
        arm=HY_RERANK_ARM,
    )
    assert decision["arm"] == HY_RERANK_ARM
    assert decision.get("candidate_source_sha256") == routed_source_sha  # type: ignore[typeddict-item]
    public_dec = public_decision_row(decision, "glm4-9b", "theoremqa", HY_RERANK_ARM)
    assert public_dec["arm"] == HY_RERANK_ARM
    assert public_dec["candidate_source_sha256"] == routed_source_sha


def test_hy_rerank_prompt_identical_given_same_candidates() -> None:
    items, corpus = _candidates_corpus(50)
    shared_ids = [cast(str, item["skill_id"]) for item in items]
    instance: JsonObject = {
        "instance_id": "theoremqa_00000",
        "dataset": "theoremqa",
        "question": "Find the residue at z = 0.",
    }
    shared_source: JsonObject = {
        "instance_id": "theoremqa_00000",
        "gold_skill_ids": ["skill_001"],
        "retrieved": [{"skill_id": sid, "score": float(50 - i)} for i, sid in enumerate(shared_ids)],
    }
    manifest = _manifest(rerank_generation(None))
    checkout = Path(__file__).resolve().parents[2] / "skill-LLM" / "external" / "SR-Agents"
    from hyskill.runtime_matched_rerank import load_native_rerank_runtime

    pinned_runtime = load_native_rerank_runtime(
        checkout,
        "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f",
    )

    always_inputs = build_rerank_inputs(
        [instance],
        {"theoremqa_00000": shared_source},
        corpus,
        pinned_runtime,
        manifest,
        "e" * 64,
        arm=RERANK_ARM,
    )
    hy_inputs = build_rerank_inputs(
        [instance],
        {"theoremqa_00000": shared_source},
        corpus,
        pinned_runtime,
        manifest,
        "e" * 64,
        arm=HY_RERANK_ARM,
    )
    assert always_inputs[0]["rendered_prompt"].encode("utf-8") == hy_inputs[0][
        "rendered_prompt"
    ].encode("utf-8")
    assert always_inputs[0]["candidate_hash"] == hy_inputs[0]["candidate_hash"]
    assert always_inputs[0]["decision_payload_hash"] == hy_inputs[0]["decision_payload_hash"]
    # Execution request hash still distinguishes the two arms
    assert always_inputs[0]["execution_request_hash"] != hy_inputs[0]["execution_request_hash"]

    pub_ans = public_answer_row(
        {
            "schema_version": "runtime-matched-baseline-evaluation-row-v1",
            "model": "glm4-9b",
            "served_model": "glm4-9b",
            "domain": "theoremqa",
            "arm": HY_RERANK_ARM,
            "instance_id": "theoremqa_00000",
            "correct": True,
            "is_validation": False,
            "failure_category": "success",
            "skill_ids_used": ["skill_001"],
            "raw_output_sha256": "1" * 64,
            "answer_payload_hash": "2" * 64,
            "execution_request_hash": "3" * 64,
            "runtime_manifest_sha256": "4" * 64,
        },
        "glm4-9b",
        "theoremqa",
        HY_RERANK_ARM,
    )
    assert pub_ans["arm"] == HY_RERANK_ARM


def test_hy_rerank_exporter_rejects_sensitive_tokens(tmp_path: Path) -> None:
    import pytest
    from scripts.export_hy_rerank_public_pack import export_hy_rerank_model_pack
    from scripts.export_runtime_matched_baselines import (
        RuntimeMatchedExportError,
        scan_pack,
    )

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "ok.json").write_text('{"arm": "hy_rerank"}\n', encoding="utf-8")
    scan_pack(clean_dir)

    leaked_dir = tmp_path / "leaked"
    leaked_dir.mkdir()
    (leaked_dir / "bad.json").write_text('{"path": "/root/secret"}\n', encoding="utf-8")
    with pytest.raises(RuntimeMatchedExportError, match="sensitive tokens"):
        scan_pack(leaked_dir)

    assert callable(export_hy_rerank_model_pack)
