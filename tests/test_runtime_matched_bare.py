from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from hyskill.runtime_matched_execution import (
    FROZEN_K2_RUNTIME_REFERENCES,
    JobBoundManifest,
    JsonObject,
    build_job_bound_manifest,
    write_json_atomic,
)
from scripts.run_runtime_matched_bare import (
    BareExecutionError,
    BareJobSummary,
    DirectEngine,
    NativeBareRuntime,
    load_native_bare_runtime,
    rendered_messages,
    run_bare_job,
)
from tests.test_runtime_matched_execution import (
    generation_fixture,
    runtime_facts_fixture,
)


RESULT_TAG: str = "deepseek7b"
SERVED_MODEL: str = "deepseek7b"


class _RequestError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int,
        body: JsonObject,
    ) -> None:
        super().__init__(message)
        self.status_code: int = status_code
        self.body: JsonObject = body


class _FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes: list[object] = list(outcomes)
        self._lock: threading.Lock = threading.Lock()
        self.calls: int = 0

    def create(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        with self._lock:
            self.calls += 1
            if not self._outcomes:
                raise AssertionError("Unexpected completion call")
            outcome: object = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=cast(str, outcome))
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            ),
        )


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions: _FakeCompletions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat: _FakeChat = _FakeChat(completions)


class _FakeEngineResult:
    def __init__(self, raw_output: str) -> None:
        self.raw_output: str = raw_output
        self.transcript: str | None = None
        self.skill_ids_used: list[str] = []
        self.meta: dict[str, object] = {}


class _FakeDirectEngine:
    def run(
        self,
        instance: JsonObject,
        skills: list[JsonObject],
        client: object,
        model: str,
    ) -> _FakeEngineResult:
        del instance, model
        assert skills == []
        typed_client: _FakeClient = cast(_FakeClient, client)
        response = typed_client.chat.completions.create()
        raw_output: str = cast(
            str,
            response.choices[0].message.content,
        )
        return _FakeEngineResult(raw_output)


def _runtime(completions: _FakeCompletions) -> NativeBareRuntime:
    def create_client(
        api_base: str | None,
        api_key: str | None,
    ) -> object:
        assert api_base == "http://127.0.0.1:8000/v1"
        assert api_key is None
        return _FakeClient(completions)

    def create_engine(
        *,
        temperature: float,
        max_tokens: int,
        thinking: bool,
    ) -> DirectEngine:
        assert temperature == 0.7
        assert max_tokens == 2048
        assert thinking is False
        return _FakeDirectEngine()

    return {
        "create_client": create_client,
        "create_engine": create_engine,
        "build_prompt": lambda instance, skills: (
            "System",
            cast(str, instance["question"]),
        ),
        "get_extra_body": lambda model, thinking: None,
        "request_error_types": (_RequestError,),
    }


def _write_job_fixture(
    tmp_path: Path,
    instances: list[JsonObject],
) -> tuple[Path, Path, Path]:
    instances_path: Path = tmp_path / "instances.json"
    corpus_path: Path = tmp_path / "corpus.json"
    code_path: Path = tmp_path / "bare_code.py"
    runtime_manifest_path: Path = tmp_path / "runtime.manifest.json"
    write_json_atomic(instances_path, instances)
    write_json_atomic(corpus_path, [])
    code_path.write_text("pass\n", encoding="utf-8")
    facts: JsonObject = runtime_facts_fixture(
        "job-theoremqa-bare",
        RESULT_TAG,
        SERVED_MODEL,
        "theoremqa",
        "bare",
        "http://127.0.0.1:8000/v1",
    )
    reference = FROZEN_K2_RUNTIME_REFERENCES[RESULT_TAG]
    checkpoint: JsonObject = cast(JsonObject, facts["checkpoint"])
    checkpoint["repository"] = reference["checkpoint_repository"]
    checkpoint["revision"] = reference["checkpoint_revision"]
    checkpoint["files_manifest_sha256"] = reference[
        "checkpoint_files_manifest_sha256"
    ]
    tokenizer: JsonObject = cast(JsonObject, facts["tokenizer"])
    tokenizer["artifacts"] = cast(
        JsonObject,
        dict(reference["tokenizer_artifacts"]),
    )
    tokenizer["chat_template_sha256"] = reference[
        "chat_template_sha256"
    ]
    endpoint: JsonObject = cast(JsonObject, facts["endpoint"])
    endpoint["vllm_version"] = reference["vllm_version"]
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
    write_json_atomic(runtime_manifest_path, manifest)
    return instances_path, corpus_path, runtime_manifest_path


def _run(
    tmp_path: Path,
    instances_path: Path,
    corpus_path: Path,
    runtime_manifest_path: Path,
    runtime: NativeBareRuntime,
    max_new_records: int,
) -> BareJobSummary:
    return run_bare_job(
        instances_path,
        corpus_path,
        runtime_manifest_path,
        tmp_path,
        tmp_path / "answers.jsonl",
        tmp_path / "usage.jsonl",
        tmp_path / "attempts.jsonl",
        RESULT_TAG,
        SERVED_MODEL,
        "http://127.0.0.1:8000/v1",
        "theoremqa",
        2,
        max_new_records,
        runtime,
        3,
        (0.0, 0.0),
    )


def _instances() -> list[JsonObject]:
    return [
        {
            "instance_id": "theoremqa_b",
            "dataset": "theoremqa",
            "question": "Question B",
        },
        {
            "instance_id": "theoremqa_a",
            "dataset": "theoremqa",
            "question": "Question A",
        },
    ]


def _jsonl_records(path: Path) -> list[JsonObject]:
    return [
        cast(JsonObject, json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_canary_resumes_without_duplicate_model_calls(
    tmp_path: Path,
) -> None:
    instances_path, corpus_path, manifest_path = _write_job_fixture(
        tmp_path,
        _instances(),
    )
    completions = _FakeCompletions(["answer-a", "answer-b"])
    runtime: NativeBareRuntime = _runtime(completions)

    canary: BareJobSummary = _run(
        tmp_path,
        instances_path,
        corpus_path,
        manifest_path,
        runtime,
        1,
    )
    assert canary["run_mode"] == "canary"
    assert canary["observed_total"] == 1
    assert canary["missing_after_run"] == 1
    assert canary["run_valid"] is True
    assert completions.calls == 1
    canary_records: list[JsonObject] = _jsonl_records(
        tmp_path / "answers.jsonl"
    )
    assert canary_records[0]["instance_id"] == "theoremqa_a"

    resumed: BareJobSummary = _run(
        tmp_path,
        instances_path,
        corpus_path,
        manifest_path,
        runtime,
        0,
    )
    assert resumed["run_mode"] == "full"
    assert resumed["observed_total"] == 2
    assert resumed["missing_after_run"] == 0
    assert resumed["run_valid"] is True
    assert resumed["reused_same_arm"] == 0
    assert completions.calls == 2

    repeated: BareJobSummary = _run(
        tmp_path,
        instances_path,
        corpus_path,
        manifest_path,
        runtime,
        0,
    )
    assert repeated["completed_this_run"] == 0
    assert repeated["observed_total"] == 2
    assert repeated["run_valid"] is True
    assert completions.calls == 2
    assert len(_jsonl_records(tmp_path / "answers.jsonl")) == 2
    assert len(_jsonl_records(tmp_path / "usage.jsonl")) == 2
    assert all(
        record["reused_same_arm"] is False
        for record in _jsonl_records(tmp_path / "answers.jsonl")
    )


def test_context_overflow_is_terminal_method_failure(
    tmp_path: Path,
) -> None:
    instances_path, corpus_path, manifest_path = _write_job_fixture(
        tmp_path,
        [_instances()[0]],
    )
    completions = _FakeCompletions(
        [
            _RequestError(
                "maximum context length exceeded",
                400,
                {"message": "prompt is too long"},
            )
        ]
    )
    summary: BareJobSummary = _run(
        tmp_path,
        instances_path,
        corpus_path,
        manifest_path,
        _runtime(completions),
        0,
    )
    assert summary["run_valid"] is True
    assert summary["unresolved"] == 0
    assert summary["failure_categories"] == {"method_failure": 1}
    assert completions.calls == 1
    record: JsonObject = _jsonl_records(tmp_path / "answers.jsonl")[0]
    assert record["failure_category"] == "method_failure"
    assert record["raw_output"] == ""
    error: JsonObject = cast(JsonObject, record["error"])
    assert error["exception_name"] == "_RequestError"
    usage: JsonObject = _jsonl_records(tmp_path / "usage.jsonl")[0]
    assert usage["status"] == "error"
    assert usage["prompt_tokens"] is None
    assert usage["usage_missing_reason"] == "request_failed:_RequestError"


def test_orphan_usage_refuses_stochastic_resampling(
    tmp_path: Path,
) -> None:
    instances_path, corpus_path, manifest_path = _write_job_fixture(
        tmp_path,
        [_instances()[0]],
    )
    completions = _FakeCompletions(["answer"])
    runtime: NativeBareRuntime = _runtime(completions)
    first: BareJobSummary = _run(
        tmp_path,
        instances_path,
        corpus_path,
        manifest_path,
        runtime,
        0,
    )
    assert first["run_valid"] is True
    (tmp_path / "answers.jsonl").unlink()
    with pytest.raises(
        BareExecutionError,
        match="Refusing to silently resample",
    ):
        _run(
            tmp_path,
            instances_path,
            corpus_path,
            manifest_path,
            runtime,
            0,
        )
    assert completions.calls == 1


def test_rendered_messages_match_native_direct_engine() -> None:
    pytest.importorskip("sragents.infer.engines.direct")
    runtime: NativeBareRuntime = load_native_bare_runtime()
    instance: JsonObject = {
        "instance_id": "theoremqa_golden",
        "dataset": "theoremqa",
        "question": "What is 1 + 1?",
        "eval_data": {
            "answer": "2",
        },
        "skill_annotations": [],
    }
    captured: list[JsonObject] = []

    class _GoldenCompletions:
        def create(self, *args: object, **kwargs: object) -> object:
            del args
            captured.append(cast(JsonObject, dict(kwargs)))
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="2")
                    )
                ]
            )

    golden_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_GoldenCompletions())
    )
    engine: DirectEngine = runtime["create_engine"](
        temperature=0.7,
        max_tokens=2048,
        thinking=False,
    )
    result = engine.run(
        instance,
        [],
        golden_client,
        "llama31-8b",
    )
    assert result.raw_output == "2"
    assert len(captured) == 1
    assert captured[0]["messages"] == cast(
        list[object],
        rendered_messages(instance, runtime),
    )
    assert captured[0]["temperature"] == 0.7
    assert captured[0]["max_tokens"] == 2048
    assert "extra_body" not in captured[0]
