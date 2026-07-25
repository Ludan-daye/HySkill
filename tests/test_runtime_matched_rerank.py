from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

import pytest

from hyskill.runtime_matched_execution import JobBoundManifest, sha256_json
from hyskill.runtime_matched_rerank import (
    RERANK_ARM,
    RERANK_DECISION_SCHEMA_VERSION,
    RETRY_DELAYS_SECONDS,
    AnswerInput,
    DirectEngine,
    EngineResult,
    JsonObject,
    NativeRerankRuntime,
    RerankInput,
    RuntimeMatchedRerankError,
    append_omitted_candidates,
    build_answer_record,
    build_rerank_decision,
    build_rerank_inputs,
    build_zero_call_answer_record,
    load_corpus,
    load_native_rerank_runtime,
    require_supported_model,
    rerank_one,
    run_answer_one,
)


class _ContextLengthError(Exception):
    def __init__(self) -> None:
        super().__init__("maximum context length exceeded")
        self.status_code: int = 400
        self.body: JsonObject = {"message": "prompt is too long"}


class _TransientError(Exception):
    def __init__(self) -> None:
        super().__init__("endpoint unavailable")
        self.status_code: int = 503
        self.body: JsonObject = {"message": "retry later"}


class _Result:
    def __init__(self, raw_output: str, skill_id: str) -> None:
        self.raw_output: str = raw_output
        self.transcript: str | None = None
        self.skill_ids_used: list[str] = [skill_id] if raw_output else []
        self.meta: JsonObject = {}


class _SequenceEngine:
    def __init__(self, outputs: list[str], skill_id: str) -> None:
        self._outputs: list[str] = list(outputs)
        self._skill_id: str = skill_id

    def run(
        self,
        instance: JsonObject,
        skills: list[JsonObject],
        client: object,
        model: str,
    ) -> EngineResult:
        assert instance["instance_id"] == "theoremqa_00000"
        assert skills[0]["skill_id"] == self._skill_id
        assert model == "served-model"
        return cast(
            EngineResult,
            _Result(self._outputs.pop(0), self._skill_id),
        )


def _parse_ranking(response: str, candidate_count: int) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for token in re.findall(r"\d+", response):
        value: int = int(token)
        if 1 <= value <= candidate_count and value not in seen:
            seen.add(value)
            output.append(value - 1)
    return output


def _manifest(generation: JsonObject) -> JobBoundManifest:
    return cast(
        JobBoundManifest,
        {
            "schema_version": "runtime-matched-job-manifest-v1",
            "runtime_facts": {},
            "generation": generation,
            "artifacts": [],
            "code_files": [],
            "code_bundle_sha256": "c" * 64,
        },
    )


def _runtime(
    responses: list[str | Exception],
) -> NativeRerankRuntime:
    response_queue: list[str | Exception] = list(responses)

    def chat(
        client: object,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
        extra_body: JsonObject | None,
    ) -> str:
        assert prompt
        assert system is None
        assert temperature == 0.0
        assert max_tokens == 1024
        assert stop is None
        value: str | Exception = response_queue.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    return {
        "prompt_template": "Q={query}\nC={candidates}",
        "format_candidates": lambda candidates: "\n".join(
            f"[{index}] {candidate['name']}: {candidate['description']}"
            for index, candidate in enumerate(candidates, start=1)
        ),
        "parse_ranking": _parse_ranking,
        "build_prompt": lambda instance, skills: (
            "system",
            cast(str, instance["question"]),
        ),
        "chat": chat,
        "create_client": lambda api_base, api_key: cast(object, object()),
        "get_extra_body": lambda model, thinking: None,
        "create_engine": cast(object, _SequenceEngine),
        "request_error_types": (_ContextLengthError, _TransientError),
        "revision": "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f",
        "source_root": "/pinned/SR-Agents/src",
    }


def _rerank_input(candidate_count: int) -> RerankInput:
    ids: list[str] = [
        f"skill_{index:02d}" for index in range(1, candidate_count + 1)
    ]
    return {
        "instance": {
            "instance_id": "theoremqa_00000",
            "dataset": "theoremqa",
            "question": "Rank skills",
        },
        "candidates": [
            {
                "skill_id": skill_id,
                "name": skill_id,
                "description": "",
                "content": skill_id,
            }
            for skill_id in ids
        ],
        "ordered_candidate_ids": ids,
        "rendered_prompt": "frozen prompt",
        "candidate_hash": "a" * 64,
        "decision_payload_hash": "b" * 64,
        "execution_request_hash": "d" * 64,
    }


def _successful_decision() -> JsonObject:
    return {
        "schema_version": RERANK_DECISION_SCHEMA_VERSION,
        "instance_id": "theoremqa_00000",
        "domain": "theoremqa",
        "arm": RERANK_ARM,
        "model": "model-tag",
        "served_model": "served-model",
        "selected_skill_id": "skill_01",
        "execution_request_hash": "d" * 64,
        "failure_category": "success",
    }


def test_native_prompt_formatter_and_parser_match_pinned_sr_agents() -> None:
    checkout: Path = (
        Path(__file__).resolve().parents[2]
        / "skill-LLM"
        / "external"
        / "SR-Agents"
    )
    runtime: NativeRerankRuntime = load_native_rerank_runtime(
        checkout,
        "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f",
    )
    candidates: list[JsonObject] = [
        {
            "skill_id": "hidden_dataset_id",
            "name": "Alpha",
            "description": "First description",
            "content": "not shown",
        },
        {
            "skill_id": "another_hidden_id",
            "name": "Beta",
            "description": "Second description",
            "content": "also not shown",
        },
    ]
    rendered: str = runtime["format_candidates"](candidates)
    assert rendered == (
        "[1] Alpha: First description\n[2] Beta: Second description"
    )
    prompt: str = runtime["prompt_template"].format(
        query="Problem text",
        candidates=rendered,
    )
    assert prompt == (
        "Given the following problem, rank the skills below by relevance. "
        "Output ONLY the skill numbers in order from most to least relevant, "
        "separated by commas.\n\n"
        "Problem:\nProblem text\n\n"
        "Skills:\n"
        "[1] Alpha: First description\n[2] Beta: Second description\n\n"
        "Most relevant first (numbers only):"
    )
    assert runtime["parse_ranking"](
        "<think>ignore 99</think>2, 1, 2",
        2,
    ) == [1, 0]


def test_load_corpus_allows_empty_content_outside_answer_selection(
    tmp_path: Path,
) -> None:
    corpus_path: Path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "web_04958",
                    "name": "Metadata-only candidate",
                    "description": "Valid rerank metadata",
                    "content": "",
                }
            ]
        ),
        encoding="utf-8",
    )

    corpus: dict[str, JsonObject] = load_corpus(corpus_path)

    assert corpus["web_04958"]["content"] == ""


def test_parse_retry_and_omitted_append_match_upstream() -> None:
    rerank_input: RerankInput = _rerank_input(6)
    runtime: NativeRerankRuntime = _runtime(["2,1", "4,3,2"])
    outcome = rerank_one(
        rerank_input,
        runtime,
        cast(object, object()),
        "served-model",
        "model-tag",
        "theoremqa",
        "job-id",
        None,
        lambda _seconds: None,
        RETRY_DELAYS_SECONDS,
    )
    assert outcome["failure_category"] == "success"
    assert outcome["parse_attempts"] == 2
    assert outcome["parse_sufficient"] is True
    assert outcome["omitted_candidate_count"] == 3
    assert outcome["reranked_candidate_ids"] == [
        "skill_04",
        "skill_03",
        "skill_02",
        "skill_01",
        "skill_05",
        "skill_06",
    ]
    assert append_omitted_candidates(
        [2, 0],
        rerank_input["ordered_candidate_ids"],
    ) == [
        "skill_03",
        "skill_01",
        "skill_02",
        "skill_04",
        "skill_05",
        "skill_06",
    ]


def test_context_failure_never_falls_back_to_bm25_top1() -> None:
    rerank_input: RerankInput = _rerank_input(50)
    outcome = rerank_one(
        rerank_input,
        _runtime([_ContextLengthError()]),
        cast(object, object()),
        "served-model",
        "model-tag",
        "theoremqa",
        "job-id",
        None,
        lambda _seconds: None,
        RETRY_DELAYS_SECONDS,
    )
    assert outcome["failure_category"] == "method_failure"
    assert outcome["client_call_attempts"] == 1
    assert outcome["reranked_candidate_ids"] == []
    assert outcome["omitted_candidate_count"] == 0
    decision = build_rerank_decision(
        rerank_input,
        outcome,
        "model-tag",
        "served-model",
        "theoremqa",
        "e" * 64,
        "c" * 64,
        "f" * 64,
    )
    assert decision["selected_skill_id"] is None


def test_transient_request_retries_then_uses_response() -> None:
    rerank_input: RerankInput = _rerank_input(6)
    outcome = rerank_one(
        rerank_input,
        _runtime(
            [
                _TransientError(),
                _TransientError(),
                "1,2,3",
            ]
        ),
        cast(object, object()),
        "served-model",
        "model-tag",
        "theoremqa",
        "job-id",
        None,
        lambda _seconds: None,
        RETRY_DELAYS_SECONDS,
    )
    assert outcome["failure_category"] == "success"
    assert outcome["client_call_attempts"] == 3
    assert outcome["reranked_candidate_ids"][0] == "skill_01"


def test_decision_failure_emits_zero_call_answer() -> None:
    instance: JsonObject = cast(
        JsonObject,
        _rerank_input(50)["instance"],
    )
    decision: JsonObject = {
        **_successful_decision(),
        "selected_skill_id": None,
        "failure_category": "method_failure",
        "error": {"exception_name": "BadRequestError"},
    }
    manifest: JobBoundManifest = _manifest(
        {
            "temperature": 0.7,
            "max_tokens": 2048,
            "thinking": False,
            "extra_body": None,
        }
    )
    record: JsonObject = build_zero_call_answer_record(
        instance,
        decision,
        "model-tag",
        "served-model",
        "theoremqa",
        manifest,
        "e" * 64,
        "f" * 64,
    )
    assert record["failure_category"] == "method_failure"
    assert record["zero_call"] is True
    assert record["engine_attempts"] == 0
    assert record["skill_ids_used"] == []


def test_direct_answer_retries_empty_and_preserves_selected_skill() -> None:
    answer_input: AnswerInput = {
        "instance": cast(JsonObject, _rerank_input(50)["instance"]),
        "skill": {
            "skill_id": "skill_01",
            "name": "Skill",
            "description": "Description",
            "content": "Content",
        },
        "messages": [{"role": "user", "content": "Question"}],
        "tools": [],
        "answer_payload_hash": "a" * 64,
        "execution_request_hash": "b" * 64,
    }
    runtime: NativeRerankRuntime = _runtime([])
    engine: DirectEngine = _SequenceEngine(
        ["", "", "final answer"],
        "skill_01",
    )
    outcome = run_answer_one(
        answer_input,
        runtime,
        engine,
        cast(object, object()),
        "served-model",
        "theoremqa",
        "job-id",
        lambda _seconds: None,
        RETRY_DELAYS_SECONDS,
    )
    assert outcome["failure_category"] == "success"
    assert outcome["engine_attempts"] == 3
    assert outcome["raw_output"] == "final answer"
    assert outcome["skill_ids_used"] == ["skill_01"]
    record = build_answer_record(
        answer_input,
        _successful_decision(),
        outcome,
        "model-tag",
        "served-model",
        "theoremqa",
        "e" * 64,
        "c" * 64,
        "f" * 64,
    )
    assert record["reused_same_arm"] is False
    assert record["expected_skill_ids"] == ["skill_01"]


def test_build_inputs_hashes_exact_prompt_and_rejects_unsupported_models() -> None:
    candidates: list[JsonObject] = [
        {
            "skill_id": f"skill_{index:02d}",
            "name": f"Name {index}",
            "description": f"Description {index}",
            "content": f"Content {index}",
        }
        for index in range(1, 51)
    ]
    corpus: dict[str, JsonObject] = {
        cast(str, candidate["skill_id"]): candidate
        for candidate in candidates
    }
    instance: JsonObject = cast(
        JsonObject,
        _rerank_input(50)["instance"],
    )
    source: JsonObject = {
        "instance_id": "theoremqa_00000",
        "gold_skill_ids": ["skill_01"],
        "retrieved": [
            {"skill_id": candidate["skill_id"], "score": float(51 - index)}
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    manifest: JobBoundManifest = _manifest(
        {
            "temperature": 0.0,
            "max_tokens": 1024,
            "thinking": False,
            "extra_body": None,
            "max_parse_attempts": 3,
            "omitted_candidate_append": True,
        }
    )
    built: list[RerankInput] = build_rerank_inputs(
        [instance],
        {"theoremqa_00000": source},
        corpus,
        _runtime([]),
        manifest,
        "e" * 64,
    )
    assert len(built) == 1
    assert len(built[0]["ordered_candidate_ids"]) == 50
    assert built[0]["candidate_hash"] == sha256_json(
        {
            "ordered_candidate_ids": built[0]["ordered_candidate_ids"],
            "formatted_candidates": "\n".join(
                f"[{index}] Name {index}: Description {index}"
                for index in range(1, 51)
            ),
        }
    )
    with pytest.raises(RuntimeMatchedRerankError, match="unavailable"):
        require_supported_model("deepseek7b")
    with pytest.raises(RuntimeMatchedRerankError, match="unavailable"):
        require_supported_model("yi15-9b")
