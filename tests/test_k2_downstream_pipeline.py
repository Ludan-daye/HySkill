import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import cast

import pytest

from hyskill.downstream_reuse import (
    CodeFileDigest,
    JsonLike,
    JsonObject,
    canonical_json,
    code_bundle_sha256_from_digests,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SR_AGENTS_SOURCE = REPOSITORY_ROOT / "external" / "SR-Agents" / "src"


class _OpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(cast(str, self.headers["content-length"]))
        request = cast(JsonObject, json.loads(self.rfile.read(length)))
        messages = cast(list[JsonObject], request["messages"])
        prompt = cast(str, messages[-1]["content"])
        content: str = (
            "2"
            if "Most relevant skill number:" in prompt
            else "Therefore, the answer is 42."
        )
        payload: JsonObject = {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": request["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        encoded: bytes = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def openai_endpoint() -> Iterator[str]:
    server: ThreadingHTTPServer = ThreadingHTTPServer(
        ("127.0.0.1", 0), _OpenAIHandler
    )
    thread: threading.Thread = threading.Thread(
        target=server.serve_forever, daemon=True
    )
    thread.start()
    try:
        host, port = cast(tuple[str, int], server.server_address)
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: JsonLike) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _run_cli_unchecked(
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    environment: dict[str, str] = dict(os.environ)
    environment["OPENAI_API_KEY"] = "EMPTY"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), str(SR_AGENTS_SOURCE)]
    )
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_cli(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    completed: subprocess.CompletedProcess[str] = _run_cli_unchecked(arguments)
    assert completed.returncode == 0, (
        f"command failed: {arguments}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return completed


def _run_cli_failure(
    arguments: Sequence[str],
    expected_error: str,
) -> subprocess.CompletedProcess[str]:
    completed: subprocess.CompletedProcess[str] = _run_cli_unchecked(arguments)
    assert completed.returncode != 0, (
        f"command unexpectedly succeeded: {arguments}\nstdout:\n{completed.stdout}"
    )
    assert expected_error in completed.stderr
    return completed


def test_runtime_manifest_builder_persists_sorted_code_members(
    tmp_path: Path,
) -> None:
    instances_path: Path = tmp_path / "instances.json"
    corpus_path: Path = tmp_path / "corpus.json"
    first_code_path: Path = tmp_path / "a.py"
    second_code_path: Path = tmp_path / "b.py"
    manifest_path: Path = tmp_path / "runtime.json"
    _write_json(instances_path, [])
    _write_json(corpus_path, [])
    first_code_path.write_text("A = 1\n", encoding="utf-8")
    second_code_path.write_text("B = 2\n", encoding="utf-8")
    _run_cli(
        [
            "scripts/build_k2_runtime_manifest.py",
            "--repository-root",
            str(tmp_path),
            "--instances",
            str(instances_path),
            "--corpus",
            str(corpus_path),
            "--model",
            "fixture",
            "--checkpoint",
            "fixture@revision",
            "--tokenizer-revision",
            "fixture",
            "--chat-template-revision",
            "fixture",
            "--served-model",
            "fixture",
            "--vllm-version",
            "fixture",
            "--dtype",
            "bfloat16",
            "--context-length",
            "8192",
            "--selector-code-files",
            str(second_code_path),
            str(first_code_path),
            "--answer-code-files",
            str(second_code_path),
            str(first_code_path),
            "--output",
            str(manifest_path),
        ]
    )
    manifest: JsonObject = cast(
        JsonObject,
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    expected_paths: list[str] = ["a.py", "b.py"]
    assert [
        cast(JsonObject, member)["path"]
        for member in cast(list[JsonLike], manifest["answer_code_files"])
    ] == expected_paths
    assert [
        cast(JsonObject, member)["path"]
        for member in cast(list[JsonLike], manifest["selector_code_files"])
    ] == expected_paths


def test_selection_reuse_answer_and_validation_pipeline(
    tmp_path: Path,
    openai_endpoint: str,
) -> None:
    instances_path: Path = tmp_path / "instances.json"
    corpus_path: Path = tmp_path / "corpus.json"
    source_path: Path = tmp_path / "routed.json"
    gated_source_path: Path = tmp_path / "gated.json"
    taus_path: Path = tmp_path / "taus.json"
    runtime_manifest_path: Path = tmp_path / "runtime.json"
    old_runtime_manifest_path: Path = tmp_path / "old-runtime.json"
    selection_path: Path = tmp_path / "selection.jsonl"
    selection_attempts_path: Path = tmp_path / "selection-attempts.jsonl"
    selected_source_path: Path = tmp_path / "selected-source.json"
    selection_validation_path: Path = tmp_path / "selection-validation.json"
    deterministic_loading_path: Path = tmp_path / "always-gated.loading.jsonl"
    select_loading_path: Path = tmp_path / "select.loading.jsonl"
    legacy_path: Path = tmp_path / "legacy.jsonl"
    audit_path: Path = tmp_path / "reuse.jsonl"
    preseed_path: Path = tmp_path / "preseed.jsonl"
    pending_path: Path = tmp_path / "pending.json"
    answers_path: Path = tmp_path / "answers.jsonl"
    answer_attempts_path: Path = tmp_path / "answer-attempts.jsonl"
    answer_validation_path: Path = tmp_path / "answer-validation.json"

    instances: list[JsonObject] = [
        {
            "instance_id": "theoremqa_00000",
            "dataset": "theoremqa",
            "question": "What is 40 + 2?",
            "skill_annotations": ["s01"],
            "eval_data": {"answer": "42", "answer_type": "integer"},
        }
    ]
    corpus: list[JsonObject] = [
        {
            "skill_id": f"s{index:02d}",
            "name": f"Skill {index:02d}",
            "description": f"Description {index:02d}",
            "content": f"Use method {index:02d}.",
        }
        for index in range(50)
    ]
    source: JsonObject = {
        "metadata": {"dataset": "theoremqa", "top_k": 50},
        "metrics": {},
        "results": [
            {
                "instance_id": "theoremqa_00000",
                "gold_skill_ids": ["s01"],
                "retrieved": [
                    {"skill_id": f"s{index:02d}", "score": 50.0 - index}
                    for index in range(50)
                ],
            }
        ],
    }
    _write_json(instances_path, instances)
    _write_json(corpus_path, corpus)
    _write_json(source_path, source)
    gated_source: JsonObject = {
        **source,
        "results": [
            {
                "instance_id": "theoremqa_00000",
                "gold_skill_ids": ["s01"],
                "retrieved": [],
            }
        ],
    }
    _write_json(gated_source_path, gated_source)
    _write_json(taus_path, {"val_ids": []})
    code_members: list[CodeFileDigest] = [
        {"path": "fixture/runtime.py", "sha256": "a" * 64}
    ]
    code_hash: str = code_bundle_sha256_from_digests(code_members)
    runtime_manifest: JsonObject = {
        "schema_version": "k2-runtime-v2",
        "instances_sha256": _sha256(instances_path),
        "corpus_sha256": _sha256(corpus_path),
        "runtime_identity": {
            "checkpoint": "local:test-model@fixture",
            "tokenizer_revision": "fixture",
            "chat_template_revision": "fixture",
            "served_model": "test-model",
            "vllm_version": "fixture",
            "dtype": "fixture",
            "context_length": 8192,
        },
        "answer_code_bundle_sha256": code_hash,
        "selector_code_bundle_sha256": code_hash,
        "answer_code_files": code_members,
        "selector_code_files": code_members,
    }
    fresh_v1_manifest: JsonObject = {
        key: value
        for key, value in runtime_manifest.items()
        if key not in {"answer_code_files", "selector_code_files"}
    }
    fresh_v1_manifest["schema_version"] = "k2-runtime-v1"
    _write_json(runtime_manifest_path, fresh_v1_manifest)
    _write_json(old_runtime_manifest_path, fresh_v1_manifest)
    legacy_path.write_text("", encoding="utf-8")

    _run_cli(
        [
            "scripts/run_select_only.py",
            "--instances",
            str(instances_path),
            "--corpus",
            str(corpus_path),
            "--source",
            str(source_path),
            "--output",
            str(selection_path),
            "--selected-source",
            str(selected_source_path),
            "--attempt-log",
            str(selection_attempts_path),
            "--runtime-manifest",
            str(runtime_manifest_path),
            "--code-bundle-sha256",
            code_hash,
            "--model",
            "test-model",
            "--api-base",
            openai_endpoint,
            "--domain",
            "theoremqa",
            "--workers",
            "1",
        ]
    )
    selection_record: JsonObject = json.loads(
        selection_path.read_text().splitlines()[0]
    )
    assert selection_record["selected_skill_id"] == "s01"
    assert selection_record["selected_rank"] == 2
    assert selection_record["failure_category"] == "success"

    _run_cli(
        [
            "scripts/validate_k2_downstream.py",
            "selection",
            "--instances",
            str(instances_path),
            "--corpus",
            str(corpus_path),
            "--source",
            str(source_path),
            "--selection",
            str(selection_path),
            "--selected-source",
            str(selected_source_path),
            "--attempt-log",
            str(selection_attempts_path),
            "--runtime-manifest",
            str(runtime_manifest_path),
            "--code-bundle-sha256",
            code_hash,
            "--model",
            "test-model",
            "--domain",
            "theoremqa",
            "--expected-count",
            "1",
            "--output",
            str(selection_validation_path),
        ]
    )
    assert json.loads(selection_validation_path.read_text())["valid"] is True

    _run_cli(
        [
            "scripts/export_k2_loading_decisions.py",
            "--instances",
            str(instances_path),
            "--always-source",
            str(source_path),
            "--gated-source",
            str(gated_source_path),
            "--taus",
            str(taus_path),
            "--model",
            "fixture-model",
            "--domain",
            "theoremqa",
            "--expected-count",
            "1",
            "--output",
            str(deterministic_loading_path),
        ]
    )
    deterministic_rows: list[JsonObject] = [
        json.loads(line)
        for line in deterministic_loading_path.read_text().splitlines()
    ]
    assert [row["arm"] for row in deterministic_rows] == [
        "routed_always",
        "routed_gated",
    ]
    assert deterministic_rows[0]["loaded"] is True
    assert deterministic_rows[1]["loaded"] is False

    _run_cli(
        [
            "scripts/export_k2_select_loading_decisions.py",
            "--instances",
            str(instances_path),
            "--selected-source",
            str(selected_source_path),
            "--selection",
            str(selection_path),
            "--taus",
            str(taus_path),
            "--model",
            "fixture-model",
            "--domain",
            "theoremqa",
            "--expected-count",
            "1",
            "--output",
            str(select_loading_path),
        ]
    )
    select_loading: JsonObject = json.loads(
        select_loading_path.read_text().splitlines()[0]
    )
    assert select_loading["expected_skill_ids"] == ["s01"]
    assert select_loading["gold_loaded"] is True

    _run_cli(
        [
            "scripts/audit_k2_reuse.py",
            "--instances",
            str(instances_path),
            "--corpus",
            str(corpus_path),
            "--decision-source",
            str(selected_source_path),
            "--legacy-jsonl",
            str(legacy_path),
            "--old-runtime-manifest",
            str(old_runtime_manifest_path),
            "--new-runtime-manifest",
            str(runtime_manifest_path),
            "--result-tag",
            "qwen3.5-4b-reference",
            "--arm",
            "routed_select",
            "--domain",
            "theoremqa",
            "--audit-output",
            str(audit_path),
            "--preseed-output",
            str(preseed_path),
            "--pending-output",
            str(pending_path),
        ]
    )
    audit_record: JsonObject = json.loads(
        audit_path.read_text().splitlines()[0]
    )
    assert audit_record["status"] == "needs_inference"
    assert audit_record["reason"] == "legacy_record_missing"

    _run_cli(
        [
            "scripts/run_k2_answers.py",
            "--instances",
            str(instances_path),
            "--corpus",
            str(corpus_path),
            "--decision-source",
            str(selected_source_path),
            "--audit",
            str(audit_path),
            "--preseed",
            str(preseed_path),
            "--output",
            str(answers_path),
            "--attempt-log",
            str(answer_attempts_path),
            "--runtime-manifest",
            str(runtime_manifest_path),
            "--model",
            "test-model",
            "--api-base",
            openai_endpoint,
            "--arm",
            "routed_select",
            "--domain",
            "theoremqa",
            "--workers",
            "1",
        ]
    )
    answer_record: JsonObject = json.loads(
        answers_path.read_text().splitlines()[0]
    )
    assert answer_record["failure_category"] == "success"
    assert answer_record["skill_ids_used"] == ["s01"]
    assert answer_record["raw_output"] == "Therefore, the answer is 42."

    _run_cli(
        [
            "scripts/validate_k2_downstream.py",
            "answer",
            "--instances",
            str(instances_path),
            "--corpus",
            str(corpus_path),
            "--decision-source",
            str(selected_source_path),
            "--answers",
            str(answers_path),
            "--audit",
            str(audit_path),
            "--legacy-jsonl",
            str(legacy_path),
            "--old-runtime-manifest",
            str(old_runtime_manifest_path),
            "--runtime-manifest",
            str(runtime_manifest_path),
            "--result-tag",
            "qwen3.5-4b-reference",
            "--model",
            "test-model",
            "--arm",
            "routed_select",
            "--domain",
            "theoremqa",
            "--expected-count",
            "1",
            "--output",
            str(answer_validation_path),
        ]
    )
    assert json.loads(answer_validation_path.read_text())["valid"] is True

    legacy_record: JsonObject = {
        **answer_record,
        "method": "select",
    }
    legacy_path.write_text(
        canonical_json(legacy_record) + "\n",
        encoding="utf-8",
    )
    bound_old_manifest: JsonObject = {
        **runtime_manifest,
        "legacy_jsonl_sha256": _sha256(legacy_path),
        "legacy_jsonl_records": 1,
        "legacy_result_tag": "qwen35-9b",
        "legacy_semantic_arm": "routed_select",
        "legacy_method_label": "select",
    }
    _write_json(runtime_manifest_path, runtime_manifest)
    _write_json(old_runtime_manifest_path, bound_old_manifest)
    bound_audit_arguments: list[str] = [
        "scripts/audit_k2_reuse.py",
        "--instances",
        str(instances_path),
        "--corpus",
        str(corpus_path),
        "--decision-source",
        str(selected_source_path),
        "--legacy-jsonl",
        str(legacy_path),
        "--old-runtime-manifest",
        str(old_runtime_manifest_path),
        "--new-runtime-manifest",
        str(runtime_manifest_path),
        "--result-tag",
        "qwen35-9b",
        "--arm",
        "routed_select",
        "--domain",
        "theoremqa",
        "--audit-output",
        str(audit_path),
        "--preseed-output",
        str(preseed_path),
        "--pending-output",
        str(pending_path),
    ]
    _run_cli(bound_audit_arguments)
    bound_audit_record: JsonObject = json.loads(
        audit_path.read_text().splitlines()[0]
    )
    assert bound_audit_record["status"] == "reused_same_arm"
    bound_validator_arguments: list[str] = [
        "scripts/validate_k2_downstream.py",
        "answer",
        "--instances",
        str(instances_path),
        "--corpus",
        str(corpus_path),
        "--decision-source",
        str(selected_source_path),
        "--answers",
        str(preseed_path),
        "--audit",
        str(audit_path),
        "--legacy-jsonl",
        str(legacy_path),
        "--old-runtime-manifest",
        str(old_runtime_manifest_path),
        "--runtime-manifest",
        str(runtime_manifest_path),
        "--result-tag",
        "qwen35-9b",
        "--model",
        "test-model",
        "--arm",
        "routed_select",
        "--domain",
        "theoremqa",
        "--expected-count",
        "1",
        "--output",
        str(answer_validation_path),
    ]
    _run_cli(bound_validator_arguments)
    assert (
        json.loads(answer_validation_path.read_text())["reused_same_arm"] == 1
    )

    _write_json(
        old_runtime_manifest_path,
        {
            **bound_old_manifest,
            "legacy_jsonl_sha256": "0" * 64,
        },
    )
    _run_cli_failure(
        bound_audit_arguments,
        "Legacy manifest evidence mismatch",
    )
    _write_json(old_runtime_manifest_path, bound_old_manifest)
    tampered_audit_record: JsonObject = {
        **bound_audit_record,
        "old_request_hash": "0" * 64,
    }
    audit_path.write_text(
        canonical_json(tampered_audit_record) + "\n",
        encoding="utf-8",
    )
    _run_cli_failure(
        bound_validator_arguments,
        "Reuse audit does not match independent recomputation",
    )
