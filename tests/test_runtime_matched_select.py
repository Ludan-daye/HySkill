from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest

from hyskill.runtime_matched_execution import (
    FROZEN_K2_RUNTIME_REFERENCES,
    FROZEN_SRAGENTS_REVISION,
    JsonLike,
    build_job_bound_manifest,
    write_json_atomic,
)
from hyskill.runtime_matched_select import (
    POOL_SIZE,
    JsonObject,
    NativeSelectorRuntime,
    SelectProtocolError,
    SelectionRecord,
    SelectorRequestFailure,
    build_selection_record,
    prepare_selection,
    require_select_eligible,
    run_selection_protocol,
    selected_skill_ids,
)
from scripts.run_runtime_matched_select_answers import (
    ANSWER_SCHEMA_VERSION,
    AnswerRuntime,
    answer_record,
    prepare_answer,
)


REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]


PROMPT_TEMPLATE: str = """\
Given the following problem, select the ONE most relevant skill. \
Output ONLY the skill number.

Problem:
{query}

Skills:
{candidates}

Most relevant skill number:"""


def corpus_fixture() -> dict[str, JsonObject]:
    """Return 50 deterministic skills with native-visible fields."""

    return {
        f"skill_{index:02d}": {
            "skill_id": f"skill_{index:02d}",
            "name": f"Skill {index:02d}",
            "description": f"Description {index:02d}",
            "content": f"Use method {index:02d}.",
        }
        for index in range(POOL_SIZE)
    }


def instance_fixture_with_id(instance_id: str) -> JsonObject:
    """Return one minimal theorem instance with an explicit ID."""

    return {
        "instance_id": instance_id,
        "dataset": "theoremqa",
        "question": "What is 40 + 2?",
        "eval_data": {"answer": "42"},
    }


def instance_fixture() -> JsonObject:
    """Return the primary minimal theorem instance."""

    return instance_fixture_with_id("theoremqa_00000")


def source_fixture_with_id(
    instance_id: str,
    score_offset: float,
) -> JsonObject:
    """Return one exact BM25 top-50 source record with an explicit ID."""

    return {
        "instance_id": instance_id,
        "gold_skill_ids": ["skill_01"],
        "retrieved": [
            {
                "skill_id": f"skill_{index:02d}",
                "score": score_offset - index,
            }
            for index in range(POOL_SIZE)
        ],
    }


def source_fixture(score_offset: float) -> JsonObject:
    """Return the primary exact BM25 top-50 source record."""

    return source_fixture_with_id("theoremqa_00000", score_offset)


def display_name(skill: JsonObject, index: int | None) -> str:
    """Mirror the native non-leaking display-name behavior."""

    raw_name = skill.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()
    return f"Skill #{index}" if index is not None else "Unnamed skill"


def format_candidates(candidates: list[JsonObject]) -> str:
    """Mirror the frozen native candidate formatter."""

    return "\n".join(
        f"[{index}] {display_name(skill, index)}: "
        f"{skill.get('description', '')}"
        for index, skill in enumerate(candidates, start=1)
    )


def parse_first_number(response: str, candidate_count: int) -> int | None:
    """Parse the first in-range integer for protocol tests."""

    for token in response.split():
        if token.isdigit() and 1 <= int(token) <= candidate_count:
            return int(token) - 1
    return None


def selector_runtime() -> NativeSelectorRuntime:
    """Return a deterministic native-selector-shaped runtime."""

    return {
        "prompt_template": PROMPT_TEMPLATE,
        "build_prompt": lambda instance, skills: (
            "",
            f"Problem:{instance['question']}\nSolution:",
        ),
        "format_candidates": format_candidates,
        "parse_first_number": parse_first_number,
        "display_name": display_name,
        "chat": cast(object, lambda *args: ""),
        "create_client": cast(object, lambda api_base, api_key: object()),
        "get_extra_body": lambda model, thinking: None,
        "request_error_types": (RuntimeError,),
    }


def selection_record_fixture(
    failure_category: str,
    selected_skill_id: str | None,
) -> SelectionRecord:
    """Return one structurally complete selection decision."""

    return cast(
        SelectionRecord,
        {
            "schema_version": "runtime-matched-select-decision-v1",
            "instance_id": "theoremqa_00000",
            "model": "glm4-9b",
            "served_model": "glm4-9b",
            "domain": "theoremqa",
            "arm": "select_bm25",
            "stage": "decision",
            "ordered_candidate_ids": [
                f"skill_{index:02d}" for index in range(POOL_SIZE)
            ],
            "candidate_hash": "1" * 64,
            "selector_payload_hash": "2" * 64,
            "execution_request_hash": "3" * 64,
            "selected_skill_id": selected_skill_id,
            "selected_rank": 1 if selected_skill_id is not None else None,
            "raw_response": "",
            "raw_responses": [],
            "parse_attempts": 0,
            "client_call_attempts": 1,
            "parse_success": False,
            "rank1_fallback": False,
            "failure_category": failure_category,
            "runtime_manifest_sha256": "4" * 64,
            "code_bundle_sha256": "5" * 64,
            "candidate_source_sha256": "6" * 64,
            "reused_same_arm": False,
            "error": None,
        },
    )


def answer_runtime() -> AnswerRuntime:
    """Return a deterministic direct-engine-shaped runtime."""

    return {
        "create_client": cast(object, lambda api_base, api_key: object()),
        "create_engine": cast(object, lambda **kwargs: object()),
        "build_prompt": lambda instance, skills: (
            "System",
            (
                f"Relevant Skill:\n{'---'.join(skills)}\n\n"
                f"Question:{instance['question']}"
                if skills
                else f"Question:{instance['question']}"
            ),
        ),
        "get_extra_body": lambda model, thinking: None,
        "request_error_types": (RuntimeError,),
    }


def test_selector_prompt_and_payload_match_native_protocol() -> None:
    """Only ordered model-visible candidates may change selector identity."""

    runtime: NativeSelectorRuntime = selector_runtime()
    corpus: dict[str, JsonObject] = corpus_fixture()
    first = prepare_selection(
        instance_fixture(),
        source_fixture(50.0),
        corpus,
        runtime,
        "glm4-9b",
        "a" * 64,
        "b" * 64,
    )
    score_changed = prepare_selection(
        instance_fixture(),
        source_fixture(500.0),
        corpus,
        runtime,
        "glm4-9b",
        "a" * 64,
        "b" * 64,
    )

    expected_candidates: str = format_candidates(list(corpus.values()))
    expected_prompt: str = PROMPT_TEMPLATE.format(
        query="Problem:What is 40 + 2?\nSolution:",
        candidates=expected_candidates,
    )
    assert first["rendered_prompt"] == expected_prompt
    assert first["generation"] == {
        "temperature": 0.0,
        "max_tokens": 64,
        "thinking": False,
        "extra_body": None,
        "max_parse_attempts": 3,
        "rank1_fallback": True,
    }
    assert first["ordered_candidate_ids"] == [
        f"skill_{index:02d}" for index in range(POOL_SIZE)
    ]
    assert first["selector_payload_hash"] == score_changed[
        "selector_payload_hash"
    ]
    assert first["execution_request_hash"] == score_changed[
        "execution_request_hash"
    ]


def test_selector_fallback_is_parse_only() -> None:
    """Three unparseable responses fall back, but request failure never does."""

    responses: Iterator[str] = iter(("none", "still none", "invalid"))
    parse_failure = run_selection_protocol(
        lambda parse_attempt: next(responses),
        parse_first_number,
        POOL_SIZE,
    )
    assert parse_failure["failure_category"] == "selector_fallback"
    assert parse_failure["selected_rank_zero_based"] == 0
    assert parse_failure["rank1_fallback"] is True
    assert parse_failure["parse_attempts"] == 3

    def context_failure(parse_attempt: int) -> str:
        raise SelectorRequestFailure(
            "method_failure",
            "BadRequestError",
            "maximum context length exceeded",
            400,
            "prompt is too long",
        )

    request_failure = run_selection_protocol(
        context_failure,
        parse_first_number,
        POOL_SIZE,
    )
    assert request_failure["failure_category"] == "method_failure"
    assert request_failure["selected_rank_zero_based"] is None
    assert request_failure["rank1_fallback"] is False
    assert request_failure["parse_attempts"] == 0


def test_selector_success_and_unsupported_models() -> None:
    """Select parses an in-range rank and rejects unsupported model tags."""

    outcome = run_selection_protocol(
        lambda parse_attempt: "17",
        parse_first_number,
        POOL_SIZE,
    )
    prepared = prepare_selection(
        instance_fixture(),
        source_fixture(50.0),
        corpus_fixture(),
        selector_runtime(),
        "glm4-9b",
        "a" * 64,
        "b" * 64,
    )
    record = build_selection_record(
        prepared,
        outcome,
        "glm4-9b",
        "glm4-9b",
        "theoremqa",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        1,
    )
    assert record["selected_rank"] == 17
    assert record["selected_skill_id"] == "skill_16"
    assert selected_skill_ids(record) == ("skill_16",)

    with pytest.raises(SelectProtocolError, match="unavailable"):
        require_select_eligible("deepseek7b")
    with pytest.raises(SelectProtocolError, match="unavailable"):
        require_select_eligible("yi15-9b")


def test_decision_failure_builds_zero_call_answer() -> None:
    """A selector method failure becomes one deterministic zero-call row."""

    decision: SelectionRecord = selection_record_fixture(
        "method_failure",
        None,
    )
    prepared = prepare_answer(
        instance_fixture(),
        decision,
        corpus_fixture(),
        answer_runtime(),
        "glm4-9b",
        "a" * 64,
        "b" * 64,
    )
    assert prepared["messages"] == []
    assert prepared["skills"] == []
    assert prepared["decision_failure"] is not None

    record = answer_record(
        prepared,
        "glm4-9b",
        "glm4-9b",
        "theoremqa",
        "",
        None,
        (),
        "method_failure",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        0,
        cast(JsonObject, prepared["decision_failure"]),
    )
    assert record["answer_call_attempts"] == 0
    assert record["failure_category"] == "method_failure"
    assert record["actual_injection_state"]["state"] == (
        "decision_failed_zero_call"
    )


def test_answer_payload_matches_native_direct_rendering() -> None:
    """A successful selection renders the exact initial direct-engine payload."""

    decision: SelectionRecord = selection_record_fixture(
        "success",
        "skill_00",
    )
    prepared = prepare_answer(
        instance_fixture(),
        decision,
        corpus_fixture(),
        answer_runtime(),
        "glm4-9b",
        "a" * 64,
        "b" * 64,
    )
    assert prepared["expected_skill_ids"] == ["skill_00"]
    assert prepared["messages"] == [
        {"role": "system", "content": "System"},
        {
            "role": "user",
            "content": (
                "Relevant Skill:\nUse method 00.\n\n"
                "Question:What is 40 + 2?"
            ),
        },
    ]
    assert prepared["generation"] == {
        "temperature": 0.7,
        "max_tokens": 2048,
        "thinking": False,
        "extra_body": None,
    }


class OpenAIHandler(BaseHTTPRequestHandler):
    """Return deterministic selector and answer completions with usage."""

    def do_POST(self) -> None:
        """Handle one OpenAI-compatible chat completion request."""

        content_length: int = int(cast(str, self.headers["content-length"]))
        request: JsonObject = cast(
            JsonObject,
            json.loads(self.rfile.read(content_length)),
        )
        messages: list[JsonObject] = cast(
            list[JsonObject],
            request["messages"],
        )
        prompt: str = cast(str, messages[-1]["content"])
        content: str = (
            "2"
            if "Most relevant skill number:" in prompt
            else "Therefore, the answer is 42."
        )
        response: JsonObject = {
            "id": "chatcmpl-fixture",
            "object": "chat.completion",
            "created": 0,
            "model": request["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 2,
                "total_tokens": 13,
            },
        }
        encoded: bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress fixture server access logs."""

        return


@pytest.fixture
def openai_endpoint() -> Iterator[str]:
    """Serve one local OpenAI-compatible endpoint."""

    server: ThreadingHTTPServer = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        OpenAIHandler,
    )
    thread: threading.Thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()
    try:
        host, port = cast(tuple[str, int], server.server_address)
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def write_fixture_module(path: Path, source: str) -> None:
    """Write one temporary fake SR-Agents module."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def write_fake_sragents(root: Path) -> Path:
    """Write the minimum native-shaped SR-Agents package used by the CLIs."""

    package_root: Path = root / "sragents"
    for relative_path in (
        "__init__.py",
        "infer/__init__.py",
        "infer/providers/__init__.py",
        "infer/engines/__init__.py",
    ):
        write_fixture_module(package_root / relative_path, "")
    write_fixture_module(
        package_root / "prompts.py",
        """
def build_prompt(instance, skills=None):
    system = "System"
    user = f"Question:{instance['question']}"
    if skills:
        user = f"Relevant Skill:\\n{'---'.join(skills)}\\n\\n{user}"
    return system, user
""".lstrip(),
    )
    write_fixture_module(
        package_root / "corpus.py",
        """
def display_name(skill, index=None):
    name = (skill.get("name") or "").strip()
    if name:
        return name
    return f"Skill #{index}" if index is not None else "Unnamed skill"
""".lstrip(),
    )
    write_fixture_module(
        package_root / "llm.py",
        """
from openai import OpenAI

def create_llm_client(api_base=None, api_key=None):
    return OpenAI(base_url=api_base, api_key=api_key or "EMPTY")

def get_extra_body(model, thinking=False):
    return None

def chat(client, model, prompt, system=None, temperature=0.7,
         max_tokens=2048, stop=None, extra_body=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
""".lstrip(),
    )
    write_fixture_module(
        package_root / "infer/providers/llm_select.py",
        """
import re

_PROMPT = \"\"\"Given the following problem, select the ONE most relevant skill. Output ONLY the skill number.

Problem:
{query}

Skills:
{candidates}

Most relevant skill number:\"\"\"

def _format_candidates(candidates):
    from sragents.corpus import display_name
    return "\\n".join(
        f"[{index}] {display_name(skill, index)}: {skill.get('description', '')}"
        for index, skill in enumerate(candidates, 1)
    )

def _parse_first_number(response, candidate_count):
    for token in re.findall(r"\\d+", response):
        value = int(token)
        if 1 <= value <= candidate_count:
            return value - 1
    return None
""".lstrip(),
    )
    write_fixture_module(
        package_root / "infer/engines/direct.py",
        """
from dataclasses import dataclass
from sragents.llm import chat
from sragents.prompts import build_prompt

@dataclass
class Result:
    raw_output: str
    transcript: str | None
    skill_ids_used: list[str]

class DirectEngine:
    def __init__(self, *, temperature, max_tokens, thinking):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, instance, skills, client, model):
        texts = [skill["content"] for skill in skills if skill.get("content")]
        system, user = build_prompt(instance, texts)
        output = chat(
            client,
            model,
            user,
            system=system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return Result(output, None, [skill["skill_id"] for skill in skills])
""".lstrip(),
    )
    write_fixture_module(
        package_root / "evaluate.py",
        """
def evaluate(raw_output, instance):
    return {
        "correct": raw_output.strip().endswith("42."),
        "extracted_answer": "42",
    }
""".lstrip(),
    )
    return root


def runtime_facts(
    api_base: str,
    job_id: str,
    stage: str,
) -> dict[str, JsonLike]:
    """Build a frozen-reference-compatible GLM runtime fixture."""

    reference = FROZEN_K2_RUNTIME_REFERENCES["glm4-9b"]
    return {
        "schema_version": "runtime-matched-runtime-facts-v1",
        "job": {
            "job_id": job_id,
            "result_tag": "glm4-9b",
            "model": "glm4-9b",
            "domain": "theoremqa",
            "arm": "select_bm25",
            "stage": stage,
        },
        "checkpoint": {
            "repository": reference["checkpoint_repository"],
            "revision": reference["checkpoint_revision"],
            "path": "/fixture/glm4-9b",
            "provenance": "verified local ModelScope mirror fixture",
            "files_manifest_sha256": reference[
                "checkpoint_files_manifest_sha256"
            ],
        },
        "tokenizer": {
            "artifacts": reference["tokenizer_artifacts"],
            "chat_template_sha256": reference["chat_template_sha256"],
        },
        "endpoint": {
            "api_base": api_base,
            "served_model": "glm4-9b",
            "process_command": "fixture-vllm",
            "vllm_version": reference["vllm_version"],
            "dtype": "bfloat16",
            "quantization": "none",
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "models_readback": {"data": [{"id": "glm4-9b"}]},
        },
        "software": {
            "python_version": "fixture",
            "pytorch_version": "fixture",
            "transformers_version": "fixture",
            "cuda_version": "fixture",
            "driver_version": "fixture",
        },
        "hardware": {
            "gpu_model": "fixture",
            "gpu_uuid": "fixture",
        },
        "source": {
            "sr_agents_revision": FROZEN_SRAGENTS_REVISION,
        },
    }


def run_select_cli(
    arguments: Sequence[str],
    fake_package_root: Path,
) -> subprocess.CompletedProcess[str]:
    """Run one Select CLI with the temporary native-shaped dependency."""

    environment: dict[str, str] = dict(os.environ)
    environment["OPENAI_API_KEY"] = "EMPTY"
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = "127.0.0.1,localhost"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(fake_package_root), str(REPOSITORY_ROOT)]
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


def test_select_cli_captures_usage_and_evaluates(
    tmp_path: Path,
    openai_endpoint: str,
) -> None:
    """Exercise decision, answer, usage, zero-reuse, and evaluation wiring."""

    fake_package_root: Path = write_fake_sragents(tmp_path / "fake")
    instances_path: Path = tmp_path / "instances.json"
    corpus_path: Path = tmp_path / "corpus.json"
    bm25_path: Path = tmp_path / "bm25.json"
    decision_manifest_path: Path = tmp_path / "decision-manifest.json"
    decision_path: Path = tmp_path / "decisions.jsonl"
    decision_attempts_path: Path = tmp_path / "decision-attempts.jsonl"
    answer_manifest_path: Path = tmp_path / "answer-manifest.json"
    answer_path: Path = tmp_path / "answers.jsonl"
    answer_attempts_path: Path = tmp_path / "answer-attempts.jsonl"
    validation_path: Path = tmp_path / "taus.json"
    evaluation_path: Path = tmp_path / "evaluation.json"
    instances: list[JsonObject] = [
        instance_fixture(),
        instance_fixture_with_id("theoremqa_00001"),
    ]
    corpus: list[JsonObject] = list(corpus_fixture().values())
    instances_path.write_text(json.dumps(instances), encoding="utf-8")
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    bm25_path.write_text(
        json.dumps(
            {
                "schema_version": "runtime-matched-bm25-v1",
                "results": [
                    source_fixture(50.0),
                    source_fixture_with_id("theoremqa_00001", 50.0),
                ],
            }
        ),
        encoding="utf-8",
    )
    decision_manifest = build_job_bound_manifest(
        runtime_facts(
            openai_endpoint,
            "glm4-theoremqa-select-decision",
            "decision",
        ),
        {
            "temperature": 0.0,
            "max_tokens": 64,
            "thinking": False,
            "extra_body": None,
        },
        (
            ("instances", instances_path),
            ("corpus", corpus_path),
            ("bm25_candidates", bm25_path),
        ),
        (
            REPOSITORY_ROOT / "hyskill/runtime_matched_execution.py",
            REPOSITORY_ROOT / "hyskill/runtime_matched_select.py",
            REPOSITORY_ROOT
            / "scripts/run_runtime_matched_select_decisions.py",
        ),
        REPOSITORY_ROOT,
    )
    write_json_atomic(decision_manifest_path, decision_manifest)
    decision_arguments: list[str] = [
        "scripts/run_runtime_matched_select_decisions.py",
        "--instances",
        str(instances_path),
        "--corpus",
        str(corpus_path),
        "--bm25-source",
        str(bm25_path),
        "--output",
        str(decision_path),
        "--attempt-log",
        str(decision_attempts_path),
        "--runtime-manifest",
        str(decision_manifest_path),
        "--repository-root",
        str(REPOSITORY_ROOT),
        "--result-tag",
        "glm4-9b",
        "--model",
        "glm4-9b",
        "--api-base",
        openai_endpoint,
        "--domain",
        "theoremqa",
        "--expected-count",
        "2",
        "--workers",
        "1",
    ]
    decision_canary = run_select_cli(
        [*decision_arguments, "--max-new-records", "1"],
        fake_package_root,
    )
    assert decision_canary.returncode == 0, (
        f"stdout:\n{decision_canary.stdout}\n"
        f"stderr:\n{decision_canary.stderr}"
    )
    canary_decision_bytes: bytes = decision_path.read_bytes()
    canary_decisions: list[JsonObject] = [
        json.loads(line)
        for line in canary_decision_bytes.decode().splitlines()
    ]
    canary_summary: JsonObject = json.loads(
        decision_canary.stdout.splitlines()[-1]
    )
    assert [row["instance_id"] for row in canary_decisions] == [
        "theoremqa_00000"
    ]
    assert canary_summary["run_mode"] == "canary"
    assert canary_summary["selected_this_run"] == 1
    assert canary_summary["missing"] == 1

    decision_full = run_select_cli(
        [*decision_arguments, "--max-new-records", "0"],
        fake_package_root,
    )
    assert decision_full.returncode == 0, (
        f"stdout:\n{decision_full.stdout}\n"
        f"stderr:\n{decision_full.stderr}"
    )
    assert decision_path.read_bytes().startswith(canary_decision_bytes)
    decisions: list[JsonObject] = [
        json.loads(line)
        for line in decision_path.read_text().splitlines()
    ]
    assert len(decisions) == 2
    assert {row["instance_id"] for row in decisions} == {
        "theoremqa_00000",
        "theoremqa_00001",
    }
    assert all(row["selected_skill_id"] == "skill_01" for row in decisions)
    assert all(row["selected_rank"] == 2 for row in decisions)
    assert all(row["stage"] == "decision" for row in decisions)
    full_decision_summary: JsonObject = json.loads(
        decision_full.stdout.splitlines()[-1]
    )
    assert full_decision_summary["run_mode"] == "full"
    assert full_decision_summary["selected_this_run"] == 1
    assert full_decision_summary["missing"] == 0
    decision_events: list[JsonObject] = [
        json.loads(line)
        for line in decision_attempts_path.read_text().splitlines()
    ]
    usage_events: list[JsonObject] = [
        event
        for event in decision_events
        if event.get("schema_version")
        == "runtime-matched-usage-event-v1"
    ]
    assert len(usage_events) == 2
    assert usage_events[0]["prompt_tokens"] == 11
    assert usage_events[0]["completion_tokens"] == 2
    assert usage_events[0]["total_tokens"] == 13

    answer_manifest = build_job_bound_manifest(
        runtime_facts(
            openai_endpoint,
            "glm4-theoremqa-select-answer",
            "answer",
        ),
        {
            "temperature": 0.7,
            "max_tokens": 2048,
            "thinking": False,
            "extra_body": None,
        },
        (
            ("instances", instances_path),
            ("corpus", corpus_path),
            ("select_decisions", decision_path),
        ),
        (
            REPOSITORY_ROOT / "hyskill/runtime_matched_execution.py",
            REPOSITORY_ROOT / "hyskill/runtime_matched_select.py",
            REPOSITORY_ROOT / "scripts/run_runtime_matched_select_answers.py",
        ),
        REPOSITORY_ROOT,
    )
    write_json_atomic(answer_manifest_path, answer_manifest)
    answer_arguments: list[str] = [
        "scripts/run_runtime_matched_select_answers.py",
        "--instances",
        str(instances_path),
        "--corpus",
        str(corpus_path),
        "--decisions",
        str(decision_path),
        "--output",
        str(answer_path),
        "--attempt-log",
        str(answer_attempts_path),
        "--runtime-manifest",
        str(answer_manifest_path),
        "--repository-root",
        str(REPOSITORY_ROOT),
        "--result-tag",
        "glm4-9b",
        "--model",
        "glm4-9b",
        "--api-base",
        openai_endpoint,
        "--domain",
        "theoremqa",
        "--expected-count",
        "2",
        "--workers",
        "1",
    ]
    answer_canary = run_select_cli(
        [*answer_arguments, "--max-new-records", "1"],
        fake_package_root,
    )
    assert answer_canary.returncode == 0, (
        f"stdout:\n{answer_canary.stdout}\n"
        f"stderr:\n{answer_canary.stderr}"
    )
    canary_answer_bytes: bytes = answer_path.read_bytes()
    canary_answers: list[JsonObject] = [
        json.loads(line)
        for line in canary_answer_bytes.decode().splitlines()
    ]
    canary_answer_summary: JsonObject = json.loads(
        answer_canary.stdout.splitlines()[-1]
    )
    assert [row["instance_id"] for row in canary_answers] == [
        "theoremqa_00000"
    ]
    assert canary_answer_summary["run_mode"] == "canary"
    assert canary_answer_summary["selected_this_run"] == 1
    assert canary_answer_summary["missing"] == 1

    answer_full = run_select_cli(
        [*answer_arguments, "--max-new-records", "0"],
        fake_package_root,
    )
    assert answer_full.returncode == 0, (
        f"stdout:\n{answer_full.stdout}\n"
        f"stderr:\n{answer_full.stderr}"
    )
    assert answer_path.read_bytes().startswith(canary_answer_bytes)
    answers: list[JsonObject] = [
        json.loads(line)
        for line in answer_path.read_text().splitlines()
    ]
    assert len(answers) == 2
    assert {row["instance_id"] for row in answers} == {
        "theoremqa_00000",
        "theoremqa_00001",
    }
    assert all(
        row["schema_version"] == ANSWER_SCHEMA_VERSION
        for row in answers
    )
    assert all(row["stage"] == "answer" for row in answers)
    assert all(row["skill_ids_used"] == ["skill_01"] for row in answers)
    assert all(
        row["raw_output"] == "Therefore, the answer is 42."
        for row in answers
    )
    assert all(row["reused_same_arm"] is False for row in answers)
    full_answer_summary: JsonObject = json.loads(
        answer_full.stdout.splitlines()[-1]
    )
    assert full_answer_summary["run_mode"] == "full"
    assert full_answer_summary["selected_this_run"] == 1
    assert full_answer_summary["missing"] == 0
    answer_events: list[JsonObject] = [
        json.loads(line)
        for line in answer_attempts_path.read_text().splitlines()
    ]
    answer_usage: list[JsonObject] = [
        event
        for event in answer_events
        if event.get("schema_version")
        == "runtime-matched-usage-event-v1"
    ]
    assert len(answer_usage) == 2
    assert answer_usage[0]["total_tokens"] == 13

    validation_path.write_text(
        json.dumps({"val_ids": []}),
        encoding="utf-8",
    )
    evaluation_result = run_select_cli(
        [
            "scripts/evaluate_runtime_matched_baselines.py",
            "--answers",
            str(answer_path),
            "--instances",
            str(instances_path),
            "--validation-source",
            str(validation_path),
            "--result-tag",
            "glm4-9b",
            "--served-model",
            "glm4-9b",
            "--domain",
            "theoremqa",
            "--arm",
            "select_bm25",
            "--expected-count",
            "2",
            "--output",
            str(evaluation_path),
        ],
        fake_package_root,
    )
    assert evaluation_result.returncode == 0, (
        f"stdout:\n{evaluation_result.stdout}\n"
        f"stderr:\n{evaluation_result.stderr}"
    )
    evaluation: JsonObject = json.loads(evaluation_path.read_text())
    details: list[JsonObject] = cast(
        list[JsonObject],
        evaluation["details"],
    )
    assert len(details) == 2
    assert all(detail["correct"] is True for detail in details)
