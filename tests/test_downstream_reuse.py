import hashlib
from pathlib import Path

import pytest

from hyskill.downstream_reuse import (
    CodeFileDigest,
    DownstreamDataError,
    allowed_legacy_label,
    answer_execution_fingerprint,
    audit_record_coverage,
    canonical_json,
    classify_request_error,
    code_bundle_sha256_from_digests,
    same_arm_preseed_eligibility,
    selector_request_fingerprint,
    sha256_file,
    validate_legacy_manifest_evidence,
    validate_runtime_manifest,
)


def test_canonical_json_uses_frozen_format() -> None:
    assert canonical_json({"z": "技能", "a": [2, 1]}) == (
        '{"a":[2,1],"z":"技能"}'
    )


def test_sha256_file_matches_standard_hash_across_multiple_chunks(
    tmp_path: Path,
) -> None:
    path: Path = tmp_path / "multi-chunk.bin"
    content: bytes = (b"0123456789abcdef" * 196_609) + b"tail"
    path.write_bytes(content)

    assert len(content) > 3 * 1024 * 1024
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_selector_hash_changes_when_candidate_identity_changes() -> None:
    generation = {
        "temperature": 0.0,
        "max_tokens": 64,
        "thinking": False,
        "extra_body": None,
        "max_parse_attempts": 3,
        "rank1_fallback": True,
    }
    common = (
        "selector-v1",
        "i1",
        {"instance_id": "i1", "question": "q"},
        "prompt",
    )
    first = selector_request_fingerprint(
        *common,
        [{"skill_id": "s1", "name": "One", "description": "d1"}],
        "c" * 64,
        {"served_model": "m"},
        generation,
        "d" * 64,
    )
    second = selector_request_fingerprint(
        *common,
        [{"skill_id": "s2", "name": "One", "description": "d1"}],
        "c" * 64,
        {"served_model": "m"},
        generation,
        "d" * 64,
    )
    assert first != second


def test_answer_hash_is_arm_scoped() -> None:
    generation = {
        "temperature": 0.7,
        "max_tokens": 2048,
        "thinking": False,
        "extra_body": None,
    }
    values = {
        "schema_version": "answer-v1",
        "instance_id": "i1",
        "instance": {"instance_id": "i1", "question": "q"},
        "messages": [{"role": "user", "content": "q"}],
        "loaded_skills": [{"skill_id": "s1", "content": "body"}],
        "tools": [],
        "instances_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "runtime_identity": {"served_model": "m"},
        "generation": generation,
        "code_bundle_sha256_value": "c" * 64,
    }
    always_hash = answer_execution_fingerprint(
        values["schema_version"],
        "routed_always",
        values["instance_id"],
        values["instance"],
        values["messages"],
        values["loaded_skills"],
        values["tools"],
        values["instances_sha256"],
        values["corpus_sha256"],
        values["runtime_identity"],
        values["generation"],
        values["code_bundle_sha256_value"],
    )
    gated_hash = answer_execution_fingerprint(
        values["schema_version"],
        "routed_gated",
        values["instance_id"],
        values["instance"],
        values["messages"],
        values["loaded_skills"],
        values["tools"],
        values["instances_sha256"],
        values["corpus_sha256"],
        values["runtime_identity"],
        values["generation"],
        values["code_bundle_sha256_value"],
    )
    assert always_hash != gated_hash


def test_same_arm_reuse_requires_every_identity_gate() -> None:
    accepted = same_arm_preseed_eligibility(
        "routed_gated",
        "routed_gated",
        "hash",
        "hash",
        "success",
        "answer",
        ("s1",),
        ("s1",),
        True,
    )
    rejected = same_arm_preseed_eligibility(
        "routed_gated",
        "routed_always",
        "hash",
        "hash",
        "success",
        "answer",
        ("s1",),
        ("s1",),
        True,
    )
    assert accepted.eligible is True
    assert rejected.eligible is False
    assert rejected.reason == "semantic_arm_mismatch"


def test_coverage_reports_duplicates_missing_and_unexpected() -> None:
    audit = audit_record_coverage(["a", "b"], ["a", "a", "c"])
    assert audit.missing_ids == ("b",)
    assert audit.duplicate_ids == ("a",)
    assert audit.unexpected_ids == ("c",)
    assert audit.complete is False


def test_runtime_manifest_accepts_tokenizer_but_rejects_credentials() -> None:
    manifest = {
        "schema_version": "runtime-v1",
        "instances_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "runtime_identity": {
            "served_model": "m",
            "tokenizer_revision": "rev",
        },
        "answer_code_bundle_sha256": "c" * 64,
        "selector_code_bundle_sha256": "d" * 64,
    }
    assert validate_runtime_manifest(
        manifest,
        "a" * 64,
        "b" * 64,
    )["runtime_identity"]["tokenizer_revision"] == "rev"
    with pytest.raises(DownstreamDataError):
        validate_runtime_manifest(
            {
                **manifest,
                "runtime_identity": {
                    **manifest["runtime_identity"],
                    "api_key": "must-not-be-persisted",
                },
            },
            "a" * 64,
            "b" * 64,
        )


def test_runtime_manifest_recomputes_sorted_code_members() -> None:
    answer_members: list[CodeFileDigest] = [
        {"path": "hyskill/a.py", "sha256": "a" * 64},
        {"path": "scripts/b.py", "sha256": "b" * 64},
    ]
    selector_members: list[CodeFileDigest] = [
        {"path": "scripts/select.py", "sha256": "c" * 64},
    ]
    manifest = {
        "schema_version": "runtime-v2",
        "instances_sha256": "d" * 64,
        "corpus_sha256": "e" * 64,
        "runtime_identity": {"served_model": "m"},
        "answer_code_bundle_sha256": code_bundle_sha256_from_digests(
            answer_members
        ),
        "selector_code_bundle_sha256": code_bundle_sha256_from_digests(
            selector_members
        ),
        "answer_code_files": answer_members,
        "selector_code_files": selector_members,
    }
    validated = validate_runtime_manifest(
        manifest,
        "d" * 64,
        "e" * 64,
    )
    assert validated["answer_code_files"] == answer_members
    with pytest.raises(DownstreamDataError):
        validate_runtime_manifest(
            {
                **manifest,
                "answer_code_files": list(reversed(answer_members)),
            },
            "d" * 64,
            "e" * 64,
        )
    with pytest.raises(DownstreamDataError):
        validate_runtime_manifest(
            {
                **manifest,
                "answer_code_bundle_sha256": "f" * 64,
            },
            "d" * 64,
            "e" * 64,
        )


def test_legacy_manifest_evidence_is_bound_to_jsonl_and_arm() -> None:
    answer_members: list[CodeFileDigest] = [
        {"path": "answer.py", "sha256": "a" * 64}
    ]
    selector_members: list[CodeFileDigest] = [
        {"path": "select.py", "sha256": "b" * 64}
    ]
    manifest = validate_runtime_manifest(
        {
            "schema_version": "runtime-v2",
            "instances_sha256": "c" * 64,
            "corpus_sha256": "d" * 64,
            "runtime_identity": {"served_model": "m"},
            "answer_code_bundle_sha256": code_bundle_sha256_from_digests(
                answer_members
            ),
            "selector_code_bundle_sha256": code_bundle_sha256_from_digests(
                selector_members
            ),
            "answer_code_files": answer_members,
            "selector_code_files": selector_members,
            "legacy_jsonl_sha256": "e" * 64,
            "legacy_jsonl_records": 10,
            "legacy_result_tag": "glm4-9b",
            "legacy_semantic_arm": "routed_gated",
            "legacy_method_label": "gated",
        },
        "c" * 64,
        "d" * 64,
    )
    assert validate_legacy_manifest_evidence(
        manifest,
        "e" * 64,
        10,
        "glm4-9b",
        "routed_gated",
        "gated",
    )["legacy_jsonl_records"] == 10
    with pytest.raises(DownstreamDataError):
        validate_legacy_manifest_evidence(
            manifest,
            "f" * 64,
            10,
            "glm4-9b",
            "routed_gated",
            "gated",
        )


def test_error_classification_keeps_method_and_infrastructure_separate() -> None:
    assert classify_request_error(
        "BadRequestError",
        "maximum context length exceeded",
        400,
        "",
    ) == "method_failure"
    assert classify_request_error(
        "RateLimitError",
        "rate limited",
        429,
        "{}",
    ) == "infra_transient"
    assert classify_request_error(
        "ValueError",
        "unknown",
        None,
        "",
    ) == "unclassified_error"


def test_qwen_legacy_labels_are_not_inferred_from_name_only() -> None:
    assert (
        allowed_legacy_label("qwen3.5-4b-reference", "routed_always")
        == "always_r"
    )
    assert (
        allowed_legacy_label("qwen3.5-4b-reference", "routed_select")
        is None
    )
    assert allowed_legacy_label("glm4-9b", "routed_select") == "select"
    assert allowed_legacy_label("deepseek7b", "routed_select") is None
    with pytest.raises(DownstreamDataError):
        allowed_legacy_label("glm4-9b-typo", "routed_select")
