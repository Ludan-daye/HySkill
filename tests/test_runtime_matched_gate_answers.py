"""Focused end-to-end test for changed Gate answer execution."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from hyskill.runtime_matched_execution import (
    FROZEN_K2_RUNTIME_REFERENCES,
    JobBoundManifest,
    JsonObject,
    build_job_bound_manifest,
    load_job_bound_manifest,
    sha256_file,
    verify_job_bound_manifest_files,
    write_json_atomic,
)
from hyskill.runtime_matched_gate import (
    GATE_DECISION_SCHEMA_VERSION,
    GATE_RERUN_ANSWER_SCHEMA_VERSION,
    GATE_RERUN_MANIFEST_SCHEMA_VERSION,
    render_direct_answer_payload,
)
from scripts.run_runtime_matched_bare import (
    DirectEngine,
    EngineResult,
    NativeBareRuntime,
)
from scripts.run_runtime_matched_gate_answers import (
    GateJobSummary,
    run_gate_job,
)
from scripts.merge_runtime_matched_gate_answers import (
    manifest_with_artifact_path_mapping,
)
from tests.test_runtime_matched_execution import (
    generation_fixture,
    runtime_facts_fixture,
)


RESULT_TAG: str = "deepseek7b"
SERVED_MODEL: str = "deepseek7b"
DOMAIN: str = "theoremqa"
ARM: str = "routed_gated"
API_BASE: str = "http://127.0.0.1:8000/v1"


class _FakeCompletions:
    """Thread-safe OpenAI-compatible completion connector."""

    def __init__(self, responses: list[str]) -> None:
        self._responses: list[str] = list(responses)
        self._lock: threading.Lock = threading.Lock()
        self.calls: int = 0

    def create(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        with self._lock:
            if not self._responses:
                raise AssertionError("Unexpected completion call")
            self.calls += 1
            response: str = self._responses.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            ),
        )


class _FakeDirectEngine:
    """Minimal native direct-engine interface for one loaded skill."""

    def run(
        self,
        instance: JsonObject,
        skills: list[JsonObject],
        client: object,
        model: str,
    ) -> EngineResult:
        del instance
        assert model == SERVED_MODEL
        assert [skill["skill_id"] for skill in skills] == ["s1"]
        typed_client = cast(object, client)
        response = typed_client.chat.completions.create()
        return cast(
            EngineResult,
            SimpleNamespace(
                raw_output=response.choices[0].message.content,
                transcript=None,
                skill_ids_used=["s1"],
                meta={},
            ),
        )


def _build_prompt(
    instance: dict[str, object],
    skills: list[str] | None,
) -> tuple[str, str]:
    """Render the deterministic direct-engine fixture prompt."""

    skill_prefix: str = "" if skills is None else skills[0] + "\n"
    return "System", skill_prefix + cast(str, instance["question"])


def _runtime(completions: _FakeCompletions) -> NativeBareRuntime:
    """Build one fake runtime with real usage instrumentation."""

    def create_client(
        api_base: str | None,
        api_key: str | None,
    ) -> object:
        assert api_base == API_BASE
        assert api_key is None
        return SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

    def create_engine(
        *,
        temperature: float,
        max_tokens: int,
        thinking: bool,
    ) -> DirectEngine:
        assert temperature == 0.7
        assert max_tokens == 2048
        assert thinking is False
        return cast(DirectEngine, _FakeDirectEngine())

    return {
        "create_client": create_client,
        "create_engine": create_engine,
        "build_prompt": _build_prompt,
        "get_extra_body": lambda model, thinking: None,
        "request_error_types": (RuntimeError,),
    }


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    """Write deterministic compact JSONL fixture rows."""

    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _freeze_runtime_facts(facts: JsonObject) -> None:
    """Apply the frozen DeepSeek runtime identity to fixture facts."""

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


def _write_gate_job_fixture(
    tmp_path: Path,
    runtime: NativeBareRuntime,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Write one changed-row Gate job and its exact manifest."""

    instances_path: Path = tmp_path / "instances.json"
    corpus_path: Path = tmp_path / "corpus.json"
    audit_path: Path = tmp_path / "audit.json"
    decisions_path: Path = tmp_path / "decisions.jsonl"
    rerun_path: Path = tmp_path / "rerun.json"
    runtime_manifest_path: Path = tmp_path / "runtime.manifest.json"
    code_path: Path = tmp_path / "gate_runner.py"
    instance: JsonObject = {
        "instance_id": "theoremqa_changed",
        "dataset": DOMAIN,
        "question": "Question",
    }
    skill: JsonObject = {
        "skill_id": "s1",
        "content": "Skill content",
    }
    rendered = render_direct_answer_payload(
        instance,
        [skill],
        SERVED_MODEL,
        runtime["build_prompt"],
        runtime["get_extra_body"],
    )
    payload_hash: str = cast(str, rendered["answer_payload_hash"])
    rerun_row: JsonObject = {
        "model": RESULT_TAG,
        "served_model": SERVED_MODEL,
        "domain": DOMAIN,
        "arm": ARM,
        "instance_id": "theoremqa_changed",
        "new_expected_skill_ids": ["s1"],
        "new_answer_payload_hash": payload_hash,
    }
    decisions: list[JsonObject] = []
    for index in range(747):
        instance_id: str = (
            "theoremqa_changed"
            if index == 0
            else f"theoremqa_other_{index:04d}"
        )
        decisions.append(
            {
                "schema_version": GATE_DECISION_SCHEMA_VERSION,
                "model": RESULT_TAG,
                "served_model": SERVED_MODEL,
                "domain": DOMAIN,
                "arm": ARM,
                "instance_id": instance_id,
                "rerun_required": index == 0,
                "expected_skill_ids": ["s1"] if index == 0 else [],
                "answer_payload_hash": payload_hash if index == 0 else "a" * 64,
            }
        )
    write_json_atomic(instances_path, [instance])
    write_json_atomic(corpus_path, [skill])
    write_json_atomic(audit_path, {"valid": True})
    _write_jsonl(decisions_path, decisions)
    write_json_atomic(
        rerun_path,
        {
            "schema_version": GATE_RERUN_MANIFEST_SCHEMA_VERSION,
            "answer_schema_version": GATE_RERUN_ANSWER_SCHEMA_VERSION,
            "model": RESULT_TAG,
            "served_model": SERVED_MODEL,
            "domain": DOMAIN,
            "arm": ARM,
            "rerun_count": 1,
            "rows": [rerun_row],
        },
    )
    code_path.write_text("pass\n", encoding="utf-8")
    facts: JsonObject = runtime_facts_fixture(
        "job-theoremqa-routed-gated",
        RESULT_TAG,
        SERVED_MODEL,
        DOMAIN,
        ARM,
        API_BASE,
    )
    _freeze_runtime_facts(facts)
    manifest: JobBoundManifest = build_job_bound_manifest(
        facts,
        generation_fixture(),
        (
            ("instances", instances_path),
            ("corpus", corpus_path),
            ("gate_audit", audit_path),
            ("gate_decisions", decisions_path),
            ("gate_rerun", rerun_path),
        ),
        (code_path,),
        tmp_path,
    )
    write_json_atomic(runtime_manifest_path, manifest)
    return (
        instances_path,
        corpus_path,
        audit_path,
        decisions_path,
        rerun_path,
        runtime_manifest_path,
    )


def _run_gate_fixture(
    tmp_path: Path,
    paths: tuple[Path, Path, Path, Path, Path, Path],
    runtime: NativeBareRuntime,
    max_new_records: int,
) -> GateJobSummary:
    """Run one fixture canary or full resume."""

    (
        instances_path,
        corpus_path,
        audit_path,
        decisions_path,
        rerun_path,
        runtime_manifest_path,
    ) = paths
    return run_gate_job(
        instances_path,
        corpus_path,
        audit_path,
        decisions_path,
        rerun_path,
        runtime_manifest_path,
        tmp_path,
        tmp_path / "answers.jsonl",
        tmp_path / "usage.jsonl",
        tmp_path / "attempts.jsonl",
        API_BASE,
        1,
        max_new_records,
        runtime,
        3,
        (0.0, 0.0),
    )


def test_changed_gate_canary_is_final_bound_and_resumable(
    tmp_path: Path,
) -> None:
    """Preserve the canary row and avoid a second stochastic request."""

    completions = _FakeCompletions(["fresh changed answer"])
    runtime: NativeBareRuntime = _runtime(completions)
    paths = _write_gate_job_fixture(tmp_path, runtime)
    canary: GateJobSummary = _run_gate_fixture(
        tmp_path,
        paths,
        runtime,
        1,
    )
    assert canary["run_mode"] == "canary"
    assert canary["observed_changed_rows"] == 1
    assert canary["missing_after_run"] == 0
    assert canary["run_valid"] is True
    assert completions.calls == 1

    resumed: GateJobSummary = _run_gate_fixture(
        tmp_path,
        paths,
        runtime,
        0,
    )
    assert resumed["run_mode"] == "full"
    assert resumed["completed_this_run"] == 0
    assert resumed["observed_changed_rows"] == 1
    assert resumed["missing_after_run"] == 0
    assert resumed["run_valid"] is True
    assert completions.calls == 1
    answers: list[JsonObject] = [
        cast(JsonObject, json.loads(line))
        for line in (tmp_path / "answers.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(answers) == 1
    assert answers[0]["failure_category"] == "success"
    assert answers[0]["expected_skill_ids"] == ["s1"]
    assert answers[0]["skill_ids_used"] == ["s1"]
    assert answers[0]["reused_same_arm"] is False


def test_runtime_manifest_artifacts_can_be_verified_after_transfer(
    tmp_path: Path,
) -> None:
    """Verify mapped copies without rewriting the producer manifest."""

    completions = _FakeCompletions(["unused"])
    paths = _write_gate_job_fixture(tmp_path, _runtime(completions))
    runtime_manifest_path: Path = paths[-1]
    manifest_sha256: str = sha256_file(runtime_manifest_path)
    manifest: JobBoundManifest = load_job_bound_manifest(
        runtime_manifest_path
    )
    transfer_root: Path = tmp_path / "transferred"
    transfer_root.mkdir()
    artifact_paths: dict[str, Path] = {}
    for artifact in manifest["artifacts"]:
        source_path: Path = Path(artifact["path"])
        destination_path: Path = transfer_root / source_path.name
        shutil.copyfile(source_path, destination_path)
        artifact_paths[artifact["name"]] = destination_path
    mapped_manifest: JobBoundManifest = (
        manifest_with_artifact_path_mapping(manifest, artifact_paths)
    )
    verify_job_bound_manifest_files(mapped_manifest, tmp_path)
    assert sha256_file(runtime_manifest_path) == manifest_sha256
    assert manifest["artifacts"] != mapped_manifest["artifacts"]
