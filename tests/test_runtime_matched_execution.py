from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from hyskill.runtime_matched_execution import (
    ExecutionContext,
    JobBoundManifest,
    JsonLike,
    JsonObject,
    JsonValue,
    RuntimeManifestError,
    answer_payload_hash,
    bind_execution_context,
    build_job_bound_manifest,
    execution_request_hash,
    load_job_bound_manifest,
    sha256_file,
    validate_frozen_k2_runtime_reference,
    verify_job_bound_manifest_files,
    write_json_atomic,
    wrap_openai_client,
)
from scripts.build_runtime_matched_runtime_manifest import main as build_manifest_main


SHA_A: str = "a" * 64
SHA_B: str = "b" * 64
SHA_C: str = "c" * 64
SHA_D: str = "d" * 64
FROZEN_SRAGENTS_REVISION: str = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"


def runtime_facts_fixture(
    job_id: str,
    result_tag: str,
    model: str,
    domain: str,
    arm: str,
    api_base: str,
) -> JsonObject:
    """Return complete credential-free runtime facts for focused tests."""

    return {
        "schema_version": "runtime-matched-runtime-facts-v1",
        "job": {
            "job_id": job_id,
            "result_tag": result_tag,
            "model": model,
            "domain": domain,
            "arm": arm,
        },
        "checkpoint": {
            "repository": "mirror/model",
            "revision": "revision-1",
            "path": "/models/model",
            "provenance": "china-mirror",
            "files_manifest_sha256": SHA_A,
        },
        "tokenizer": {
            "artifacts": {
                "tokenizer.json": SHA_B,
                "tokenizer_config.json": SHA_C,
            },
            "chat_template_sha256": SHA_D,
        },
        "endpoint": {
            "api_base": api_base,
            "served_model": model,
            "process_command": "vllm serve /models/model",
            "vllm_version": "0.19.1",
            "dtype": "bfloat16",
            "quantization": "none",
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "models_readback": {
                "data": [
                    {
                        "id": model,
                    }
                ]
            },
        },
        "software": {
            "python_version": "3.10.12",
            "pytorch_version": "2.10.0",
            "transformers_version": "5.13.1",
            "cuda_version": "12.8",
            "driver_version": "570.00",
        },
        "hardware": {
            "gpu_model": "NVIDIA A100",
            "gpu_uuid": "GPU-test",
        },
        "source": {
            "sr_agents_revision": FROZEN_SRAGENTS_REVISION,
        },
    }


def generation_fixture() -> JsonObject:
    """Return the frozen direct-answer generation object."""

    return {
        "temperature": 0.7,
        "max_tokens": 2048,
        "thinking": False,
        "extra_body": None,
    }


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self._responses: list[object] = list(responses)
        self._lock: threading.Lock = threading.Lock()

    def create(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        with self._lock:
            if not self._responses:
                raise AssertionError("Unexpected completion call")
            response: object = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions: _FakeCompletions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat: _FakeChat = _FakeChat(completions)


class _RequestError(Exception):
    def __init__(self, message: str, status_code: int, body: JsonObject) -> None:
        super().__init__(message)
        self.status_code: int = status_code
        self.body: JsonObject = body


def _response(
    prompt_tokens: int,
    completion_tokens: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    )


def test_payload_hash_is_separate_from_execution_identity() -> None:
    instance: JsonObject = {
        "instance_id": "theoremqa_00001",
        "dataset": "theoremqa",
        "question": "Question",
    }
    messages: list[JsonObject] = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question"},
    ]
    payload_hash: str = answer_payload_hash(
        "runtime-matched-answer-payload-v1",
        instance,
        messages,
        [],
        [],
        generation_fixture(),
    )
    repeated_payload_hash: str = answer_payload_hash(
        "runtime-matched-answer-payload-v1",
        instance,
        messages,
        [],
        [],
        generation_fixture(),
    )
    first_execution_hash: str = execution_request_hash(
        "runtime-matched-answer-payload-v1",
        payload_hash,
        SHA_A,
        SHA_B,
    )
    second_execution_hash: str = execution_request_hash(
        "runtime-matched-answer-payload-v1",
        payload_hash,
        SHA_C,
        SHA_D,
    )
    assert payload_hash == repeated_payload_hash
    assert first_execution_hash != second_execution_hash


def test_usage_capture_is_context_local_and_numbers_subcalls() -> None:
    responses: list[object] = [_response(10, 2) for _index in range(4)]
    client = _FakeClient(_FakeCompletions(responses))
    events: list[JsonObject] = []
    event_lock: threading.Lock = threading.Lock()

    def sink(event: dict[str, JsonLike] | JsonObject) -> None:
        with event_lock:
            events.append(cast(JsonObject, dict(event)))

    wrapped = wrap_openai_client(client, sink)

    def worker(instance_id: str, payload_sha: str, execution_sha: str) -> None:
        context = ExecutionContext(
            "job-1",
            "model-1",
            "theoremqa",
            "bare",
            instance_id,
            1,
            payload_sha,
            execution_sha,
        )
        with bind_execution_context(context):
            wrapped.chat.completions.create(model="model-1")
            wrapped.chat.completions.create(model="model-1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(worker, "instance-a", SHA_A, SHA_B),
            executor.submit(worker, "instance-b", SHA_C, SHA_D),
        ]
        for future in futures:
            future.result()

    assert len(events) == 4
    by_instance: dict[str, list[JsonObject]] = {}
    for event in events:
        instance_id: str = cast(str, event["instance_id"])
        by_instance.setdefault(instance_id, []).append(event)
    assert sorted(by_instance) == ["instance-a", "instance-b"]
    assert {
        instance_id: sorted(
            cast(int, event["http_subcall"]) for event in instance_events
        )
        for instance_id, instance_events in by_instance.items()
    } == {
        "instance-a": [1, 2],
        "instance-b": [1, 2],
    }
    assert {
        instance_id: {
            cast(str, event["answer_payload_hash"])
            for event in instance_events
        }
        for instance_id, instance_events in by_instance.items()
    } == {
        "instance-a": {SHA_A},
        "instance-b": {SHA_C},
    }


def test_usage_capture_preserves_response_and_records_missing_or_error() -> None:
    response_without_usage = SimpleNamespace(result="same-object")
    request_error = _RequestError(
        "endpoint unavailable",
        503,
        {"message": "temporary"},
    )
    client = _FakeClient(
        _FakeCompletions([response_without_usage, request_error])
    )
    events: list[JsonObject] = []
    wrapped = wrap_openai_client(
        client,
        lambda event: events.append(cast(JsonObject, dict(event))),
    )
    context = ExecutionContext(
        "job-1",
        "model-1",
        "theoremqa",
        "bare",
        "instance-a",
        1,
        SHA_A,
        SHA_B,
    )
    with bind_execution_context(context):
        observed = wrapped.chat.completions.create(model="model-1")
        with pytest.raises(_RequestError):
            wrapped.chat.completions.create(model="model-1")
    assert observed is response_without_usage
    assert events[0]["prompt_tokens"] is None
    assert events[0]["completion_tokens"] is None
    assert events[0]["total_tokens"] is None
    assert events[0]["usage_missing_reason"] == "response_usage_absent"
    assert events[1]["status"] == "error"
    assert events[1]["usage_missing_reason"] == "request_failed:_RequestError"
    assert events[1]["status_code"] == 503


def test_manifest_validates_files_and_rejects_credentials(
    tmp_path: Path,
) -> None:
    instances_path: Path = tmp_path / "instances.json"
    corpus_path: Path = tmp_path / "corpus.json"
    code_path: Path = tmp_path / "runner.py"
    instances_path.write_text("[]\n", encoding="utf-8")
    corpus_path.write_text("[]\n", encoding="utf-8")
    code_path.write_text("pass\n", encoding="utf-8")
    facts: JsonObject = runtime_facts_fixture(
        "job-1",
        "result-tag",
        "model-1",
        "theoremqa",
        "bare",
        "http://127.0.0.1:8000/v1",
    )
    manifest: JobBoundManifest = build_job_bound_manifest(
        facts,
        generation_fixture(),
        (
            ("instances", instances_path),
            ("corpus", corpus_path),
        ),
        (code_path,),
        tmp_path,
    )
    verify_job_bound_manifest_files(manifest, tmp_path)
    assert manifest["artifacts"][0]["name"] == "corpus"
    assert manifest["artifacts"][1]["name"] == "instances"
    assert manifest["code_files"][0]["path"] == "runner.py"
    assert manifest["artifacts"][1]["sha256"] == sha256_file(instances_path)

    instances_path.write_text("[{}]\n", encoding="utf-8")
    with pytest.raises(RuntimeManifestError, match="identity mismatch"):
        verify_job_bound_manifest_files(manifest, tmp_path)

    secret_facts: JsonObject = runtime_facts_fixture(
        "job-2",
        "result-tag",
        "model-1",
        "theoremqa",
        "bare",
        "http://127.0.0.1:8000/v1",
    )
    secret_facts["password"] = "must-not-enter-manifest"
    with pytest.raises(RuntimeManifestError, match="credential-like"):
        build_job_bound_manifest(
            secret_facts,
            generation_fixture(),
            (("corpus", corpus_path),),
            (code_path,),
            tmp_path,
        )


def test_frozen_runtime_reference_rejects_endpoint_drift() -> None:
    facts: JsonObject = runtime_facts_fixture(
        "job-deepseek",
        "deepseek7b",
        "deepseek7b",
        "theoremqa",
        "bare",
        "http://127.0.0.1:8000/v1",
    )
    checkpoint: JsonObject = cast(JsonObject, facts["checkpoint"])
    checkpoint.update(
        {
            "repository": "deepseek-ai/deepseek-llm-7b-chat",
            "revision": "snapshots/master",
            "files_manifest_sha256": (
                "25b7f08040a12a38ed6a4fdca625063e18091926a30813d56a3c87e3cbe1f03b"
            ),
        }
    )
    tokenizer: JsonObject = cast(JsonObject, facts["tokenizer"])
    tokenizer.update(
        {
            "artifacts": {
                "tokenizer.json": (
                    "a08b02921f08548065a7b2ec13b2ffeed873231add60f9c3c7b08b04f2cc212a"
                ),
                "tokenizer_config.json": (
                    "9e4d4a34afe6db6096508a5363b065cf684ec3a9047da1c2dbe30bd8537a6086"
                ),
            },
            "chat_template_sha256": (
                "9e4d4a34afe6db6096508a5363b065cf684ec3a9047da1c2dbe30bd8537a6086"
            ),
        }
    )
    validate_frozen_k2_runtime_reference(facts)
    endpoint: JsonObject = cast(JsonObject, facts["endpoint"])
    endpoint["max_model_len"] = 4096
    with pytest.raises(RuntimeManifestError, match="max_model_len"):
        validate_frozen_k2_runtime_reference(facts)


def test_manifest_cli_builds_and_reloads_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_facts_path: Path = tmp_path / "runtime-facts.json"
    generation_path: Path = tmp_path / "generation.json"
    instances_path: Path = tmp_path / "instances.json"
    code_path: Path = tmp_path / "runner.py"
    output_path: Path = tmp_path / "runtime.manifest.json"
    write_json_atomic(
        runtime_facts_path,
        runtime_facts_fixture(
            "job-1",
            "result-tag",
            "model-1",
            "theoremqa",
            "bare",
            "http://127.0.0.1:8000/v1",
        ),
    )
    write_json_atomic(generation_path, generation_fixture())
    write_json_atomic(instances_path, [])
    code_path.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_runtime_matched_runtime_manifest.py",
            "--runtime-facts",
            str(runtime_facts_path),
            "--generation",
            str(generation_path),
            "--artifact",
            f"instances={instances_path}",
            "--code-file",
            "runner.py",
            "--repository-root",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )
    build_manifest_main()
    manifest: JobBoundManifest = load_job_bound_manifest(output_path)
    assert manifest["artifacts"][0]["name"] == "instances"
    assert manifest["code_files"][0]["path"] == "runner.py"
    summary: JsonObject = cast(
        JsonObject,
        json.loads(capsys.readouterr().out),
    )
    assert summary["event"] == "runtime_matched_runtime_manifest_built"
    assert summary["runtime_manifest_sha256"] == sha256_file(output_path)
