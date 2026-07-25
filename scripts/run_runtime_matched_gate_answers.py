#!/usr/bin/env python3
"""Run only Fresh-Bare-induced changed K=2 Gate answer payloads."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict, cast

from hyskill.runtime_matched_execution import (
    ExecutionContext,
    FailureCategory,
    JobBoundManifest,
    JsonLike,
    JsonObject,
    JsonValue,
    OpenAIClientLike,
    RuntimeManifestError,
    UsageSink,
    bind_execution_context,
    canonical_json,
    classify_request_error,
    error_context,
    execution_request_hash,
    load_job_bound_manifest,
    load_json_file,
    require_json_object,
    require_nonempty_string,
    sha256_file,
    validate_frozen_k2_runtime_reference,
    verify_job_bound_manifest_files,
    wrap_openai_client,
)
from hyskill.runtime_matched_gate import (
    ANSWER_MAX_TOKENS,
    ANSWER_PAYLOAD_SCHEMA_VERSION,
    ANSWER_TEMPERATURE,
    ANSWER_THINKING,
    GATE_DECISION_SCHEMA_VERSION,
    GATE_RERUN_ANSWER_SCHEMA_VERSION,
    GATE_RERUN_MANIFEST_SCHEMA_VERSION,
    RULE_DOMAIN_COUNTS,
    GateArm,
    JsonObject as GateJsonObject,
    RuntimeMatchedGateError,
    index_corpus,
    index_instances,
    render_direct_answer_payload,
    require_boolean,
    require_gate_arm,
    require_list,
    require_object,
    require_string,
    require_string_list,
)
from scripts.merge_runtime_matched_gate_answers import (
    validate_rerun_answer,
)
from scripts.run_runtime_matched_bare import (
    DirectEngine,
    EngineResult,
    JsonlLine,
    JsonlUsageSink,
    NativeBareRuntime,
    append_jsonl,
    attempt_payload,
    indexed_lines,
    load_jsonl,
    load_native_bare_runtime,
    required_nested_object,
    validate_failure_category,
    validate_named_artifact,
)


MAX_ENGINE_ATTEMPTS: int = 3
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0)
REQUIRED_ARTIFACT_NAMES: frozenset[str] = frozenset(
    {
        "instances",
        "corpus",
        "gate_audit",
        "gate_decisions",
        "gate_rerun",
    }
)


class PreparedGateRequest(TypedDict):
    """One validated changed Gate request."""

    instance: JsonObject
    loaded_skills: list[JsonObject]
    expected_skill_ids: list[str]
    answer_payload_hash: str
    execution_request_hash: str
    rerun_row: GateJsonObject


class GateRunOutcome(TypedDict):
    """One terminal changed Gate answer."""

    record: JsonObject
    engine_attempts: int


class GateJobSummary(TypedDict):
    """Machine-readable changed Gate job state."""

    event: str
    job_id: str
    model: str
    served_model: str
    domain: str
    arm: str
    run_mode: str
    expected_changed_rows: int
    observed_changed_rows: int
    pending_before_run: int
    selected_this_run: int
    completed_this_run: int
    unresolved: int
    missing_after_run: int
    failure_categories: dict[str, int]
    runtime_manifest_sha256: str
    output: str
    output_sha256: str
    run_valid: bool


def parse_args() -> argparse.Namespace:
    """Parse one explicit changed Gate answer job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--gate-audit", required=True, type=Path)
    parser.add_argument("--gate-decisions", required=True, type=Path)
    parser.add_argument("--gate-rerun", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--usage-log", required=True, type=Path)
    parser.add_argument("--attempt-log", required=True, type=Path)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--max-new-records", required=True, type=int)
    return parser.parse_args()


def load_rerun_manifest(path: Path) -> GateJsonObject:
    """Load one non-empty changed-row manifest."""

    payload: GateJsonObject = require_object(
        load_json_file(path, "gate-rerun"),
        "gate-rerun",
    )
    if payload.get("schema_version") != GATE_RERUN_MANIFEST_SCHEMA_VERSION:
        raise RuntimeMatchedGateError(
            "Gate rerun manifest schema mismatch: "
            f"path={path}, schema={payload.get('schema_version')!r}"
        )
    if payload.get("answer_schema_version") != GATE_RERUN_ANSWER_SCHEMA_VERSION:
        raise RuntimeMatchedGateError(
            "Gate rerun answer schema mismatch: "
            f"path={path}, schema={payload.get('answer_schema_version')!r}"
        )
    rerun_count: JsonValue | None = payload.get("rerun_count")
    if (
        isinstance(rerun_count, bool)
        or not isinstance(rerun_count, int)
        or rerun_count <= 0
    ):
        raise RuntimeMatchedGateError(
            "Changed Gate runner requires a positive rerun count: "
            f"path={path}, rerun_count={rerun_count!r}"
        )
    rows: list[JsonValue] = require_list(payload.get("rows"), "gate-rerun.rows")
    if len(rows) != rerun_count:
        raise RuntimeMatchedGateError(
            "Gate rerun row count mismatch: "
            f"path={path}, declared={rerun_count}, observed={len(rows)}"
        )
    return payload


def index_rerun_rows(payload: Mapping[str, JsonValue]) -> dict[str, GateJsonObject]:
    """Index and validate all rerun-required row identities."""

    model: str = require_string(payload.get("model"), "gate-rerun.model")
    served_model: str = require_string(
        payload.get("served_model"),
        "gate-rerun.served_model",
    )
    domain: str = require_string(payload.get("domain"), "gate-rerun.domain")
    arm: GateArm = require_gate_arm(
        require_string(payload.get("arm"), "gate-rerun.arm")
    )
    output: dict[str, GateJsonObject] = {}
    rows: list[JsonValue] = require_list(payload.get("rows"), "gate-rerun.rows")
    for index, value in enumerate(rows):
        row: GateJsonObject = require_object(value, f"gate-rerun.rows[{index}]")
        instance_id: str = require_string(
            row.get("instance_id"),
            f"gate-rerun.rows[{index}].instance_id",
        )
        if instance_id in output:
            raise RuntimeMatchedGateError(
                f"Gate rerun contains a duplicate ID: instance_id={instance_id}"
            )
        expected_fields: tuple[tuple[str, JsonValue], ...] = (
            ("model", model),
            ("served_model", served_model),
            ("domain", domain),
            ("arm", arm),
        )
        mismatches: list[str] = [
            f"{field}:expected={expected!r},observed={row.get(field)!r}"
            for field, expected in expected_fields
            if row.get(field) != expected
        ]
        require_string_list(
            row.get("new_expected_skill_ids"),
            f"gate-rerun:{instance_id}.new_expected_skill_ids",
        )
        require_nonempty_string(
            row.get("new_answer_payload_hash"),
            f"gate-rerun:{instance_id}.new_answer_payload_hash",
        )
        if mismatches:
            raise RuntimeMatchedGateError(
                "Gate rerun row identity mismatch: "
                f"instance_id={instance_id}, mismatches={mismatches}"
            )
        output[instance_id] = row
    return output


def index_decisions(
    path: Path,
    model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
) -> dict[str, GateJsonObject]:
    """Load exact-domain Gate decisions by unique instance ID."""

    lines: list[JsonlLine] = load_jsonl(path, "gate-decisions", False)
    expected_count: int | None = RULE_DOMAIN_COUNTS.get(domain)
    if expected_count is None or len(lines) != expected_count:
        raise RuntimeMatchedGateError(
            "Gate decision denominator mismatch: "
            f"domain={domain}, expected={expected_count}, observed={len(lines)}"
        )
    output: dict[str, GateJsonObject] = {}
    for index, line in enumerate(lines):
        row: GateJsonObject = cast(GateJsonObject, line["record"])
        instance_id: str = require_string(
            row.get("instance_id"),
            f"gate-decisions[{index}].instance_id",
        )
        if instance_id in output:
            raise RuntimeMatchedGateError(
                f"Gate decisions contain a duplicate ID: instance_id={instance_id}"
            )
        expected_fields: tuple[tuple[str, JsonValue], ...] = (
            ("schema_version", GATE_DECISION_SCHEMA_VERSION),
            ("model", model),
            ("served_model", served_model),
            ("domain", domain),
            ("arm", arm),
        )
        mismatches: list[str] = [
            f"{field}:expected={expected!r},observed={row.get(field)!r}"
            for field, expected in expected_fields
            if row.get(field) != expected
        ]
        if mismatches:
            raise RuntimeMatchedGateError(
                "Gate decision identity mismatch: "
                f"instance_id={instance_id}, mismatches={mismatches}"
            )
        output[instance_id] = row
    return output


def manifest_artifact_names(manifest: JobBoundManifest) -> set[str]:
    """Return the exact named artifact inventory."""

    names: list[str] = []
    for index, raw_artifact in enumerate(manifest["artifacts"]):
        name: str = require_nonempty_string(
            raw_artifact.get("name"),
            f"manifest.artifacts[{index}].name",
        )
        names.append(name)
    if len(names) != len(set(names)):
        raise RuntimeManifestError(
            f"Runtime manifest contains duplicate artifact names: names={names}"
        )
    return set(names)


def validate_gate_runtime_manifest(
    manifest: JobBoundManifest,
    repository_root: Path,
    input_paths: Mapping[str, Path],
    rerun_manifest: Mapping[str, JsonValue],
    api_base: str,
    runtime: NativeBareRuntime,
) -> tuple[str, str]:
    """Validate runtime, protocol, code, and every changed-job input."""

    verify_job_bound_manifest_files(manifest, repository_root)
    validate_frozen_k2_runtime_reference(manifest["runtime_facts"])
    model: str = require_string(rerun_manifest.get("model"), "gate-rerun.model")
    served_model: str = require_string(
        rerun_manifest.get("served_model"),
        "gate-rerun.served_model",
    )
    domain: str = require_string(
        rerun_manifest.get("domain"),
        "gate-rerun.domain",
    )
    arm: GateArm = require_gate_arm(
        require_string(rerun_manifest.get("arm"), "gate-rerun.arm")
    )
    job: JsonObject = required_nested_object(
        manifest["runtime_facts"],
        "job",
        "runtime_facts",
    )
    endpoint: JsonObject = required_nested_object(
        manifest["runtime_facts"],
        "endpoint",
        "runtime_facts",
    )
    expected_fields: tuple[tuple[str, JsonValue, JsonValue | None], ...] = (
        ("job.result_tag", model, job.get("result_tag")),
        ("job.model", served_model, job.get("model")),
        ("job.domain", domain, job.get("domain")),
        ("job.arm", arm, job.get("arm")),
        ("endpoint.api_base", api_base, endpoint.get("api_base")),
        ("endpoint.served_model", served_model, endpoint.get("served_model")),
        ("endpoint.dtype", "bfloat16", endpoint.get("dtype")),
        ("endpoint.quantization", "none", endpoint.get("quantization")),
        ("endpoint.max_model_len", 8192, endpoint.get("max_model_len")),
    )
    mismatches: list[str] = [
        f"{field}:expected={expected!r},observed={observed!r}"
        for field, expected, observed in expected_fields
        if observed != expected
    ]
    expected_generation: JsonObject = {
        "temperature": ANSWER_TEMPERATURE,
        "max_tokens": ANSWER_MAX_TOKENS,
        "thinking": ANSWER_THINKING,
        "extra_body": cast(
            JsonValue,
            runtime["get_extra_body"](served_model, ANSWER_THINKING),
        ),
    }
    if manifest["generation"] != expected_generation:
        mismatches.append(
            "generation:"
            f"expected={expected_generation!r},observed={manifest['generation']!r}"
        )
    if mismatches:
        raise RuntimeManifestError(
            "Changed Gate runtime manifest mismatch: "
            f"mismatches={mismatches}"
        )
    observed_artifact_names: set[str] = manifest_artifact_names(manifest)
    if observed_artifact_names != REQUIRED_ARTIFACT_NAMES:
        raise RuntimeManifestError(
            "Changed Gate artifact inventory mismatch: "
            f"expected={sorted(REQUIRED_ARTIFACT_NAMES)}, "
            f"observed={sorted(observed_artifact_names)}"
        )
    for name, path in input_paths.items():
        validate_named_artifact(manifest, name, path)
    return (
        validate_named_artifact(manifest, "instances", input_paths["instances"]),
        validate_named_artifact(manifest, "corpus", input_paths["corpus"]),
    )


def loaded_skills(
    expected_skill_ids: Sequence[str],
    corpus: Mapping[str, JsonObject],
    instance_id: str,
) -> list[JsonObject]:
    """Resolve one ordered changed Gate injection."""

    output: list[JsonObject] = []
    for skill_id in expected_skill_ids:
        skill: JsonObject | None = corpus.get(skill_id)
        if skill is None:
            raise RuntimeMatchedGateError(
                "Changed Gate row references a missing skill: "
                f"instance_id={instance_id}, skill_id={skill_id}"
            )
        output.append(skill)
    return output


def prepare_requests(
    rerun_rows: Mapping[str, GateJsonObject],
    decisions: Mapping[str, GateJsonObject],
    instances: Mapping[str, JsonObject],
    corpus: Mapping[str, JsonObject],
    served_model: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    runtime: NativeBareRuntime,
) -> dict[str, PreparedGateRequest]:
    """Render and hash every changed Gate request before inference."""

    missing_instances: list[str] = sorted(set(rerun_rows) - set(instances))
    if missing_instances:
        raise RuntimeMatchedGateError(
            f"Gate rerun IDs are absent from instances: sample={missing_instances[:20]}"
        )
    output: dict[str, PreparedGateRequest] = {}
    for instance_id, rerun_row in rerun_rows.items():
        decision: GateJsonObject | None = decisions.get(instance_id)
        if decision is None:
            raise RuntimeMatchedGateError(
                f"Gate rerun ID has no decision row: instance_id={instance_id}"
            )
        if require_boolean(
            decision.get("rerun_required"),
            f"gate-decision:{instance_id}.rerun_required",
        ) is not True:
            raise RuntimeMatchedGateError(
                f"Gate rerun decision is not marked for rerun: instance_id={instance_id}"
            )
        expected_skill_ids: list[str] = require_string_list(
            rerun_row.get("new_expected_skill_ids"),
            f"gate-rerun:{instance_id}.new_expected_skill_ids",
        )
        decision_skills: list[str] = require_string_list(
            decision.get("expected_skill_ids"),
            f"gate-decision:{instance_id}.expected_skill_ids",
        )
        if decision_skills != expected_skill_ids:
            raise RuntimeMatchedGateError(
                "Gate rerun and decision skill injections differ: "
                f"instance_id={instance_id}, rerun={expected_skill_ids}, "
                f"decision={decision_skills}"
            )
        skill_objects: list[JsonObject] = loaded_skills(
            expected_skill_ids,
            corpus,
            instance_id,
        )
        rendered = render_direct_answer_payload(
            instances[instance_id],
            skill_objects,
            served_model,
            runtime["build_prompt"],
            runtime["get_extra_body"],
        )
        expected_payload_hash: str = require_string(
            rerun_row.get("new_answer_payload_hash"),
            f"gate-rerun:{instance_id}.new_answer_payload_hash",
        )
        if rendered["answer_payload_hash"] != expected_payload_hash:
            raise RuntimeMatchedGateError(
                "Changed Gate payload hash is stale: "
                f"instance_id={instance_id}, expected={expected_payload_hash}, "
                f"observed={rendered['answer_payload_hash']}"
            )
        if decision.get("answer_payload_hash") != expected_payload_hash:
            raise RuntimeMatchedGateError(
                "Gate decision payload hash differs from rerun manifest: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = {
            "instance": instances[instance_id],
            "loaded_skills": skill_objects,
            "expected_skill_ids": expected_skill_ids,
            "answer_payload_hash": expected_payload_hash,
            "execution_request_hash": execution_request_hash(
                ANSWER_PAYLOAD_SCHEMA_VERSION,
                expected_payload_hash,
                runtime_manifest_sha256,
                code_bundle_sha256,
            ),
            "rerun_row": rerun_row,
        }
    return output


def validate_existing_outputs(
    lines: Sequence[JsonlLine],
    prepared: Mapping[str, PreparedGateRequest],
    model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
) -> dict[str, JsonlLine]:
    """Validate all final-bound changed answer rows before resume."""

    indexed: dict[str, JsonlLine] = indexed_lines(lines, "gate-answer-output")
    unexpected_ids: list[str] = sorted(set(indexed) - set(prepared))
    if unexpected_ids:
        raise RuntimeMatchedGateError(
            "Changed Gate output contains unexpected IDs: "
            f"sample={unexpected_ids[:20]}"
        )
    for instance_id, line in indexed.items():
        validate_rerun_answer(
            line["record"],
            prepared[instance_id]["rerun_row"],
            model,
            served_model,
            domain,
            arm,
            runtime_manifest_sha256,
            code_bundle_sha256,
        )
    return indexed


def validate_usage_journal(
    lines: Sequence[JsonlLine],
    output_index: Mapping[str, JsonlLine],
    prepared: Mapping[str, PreparedGateRequest],
    job_id: str,
    served_model: str,
    domain: str,
    arm: GateArm,
) -> None:
    """Reject stale, duplicate, or orphan changed-job usage events."""

    event_ids: set[tuple[str, int, int]] = set()
    usage_ids: set[str] = set()
    for index, line in enumerate(lines):
        event: JsonObject = line["record"]
        instance_id: str = require_nonempty_string(
            event.get("instance_id"),
            f"usage[{index}].instance_id",
        )
        prepared_row: PreparedGateRequest | None = prepared.get(instance_id)
        if prepared_row is None:
            raise RuntimeMatchedGateError(
                f"Usage event is outside changed IDs: instance_id={instance_id}"
            )
        logical_attempt: JsonValue | None = event.get("logical_attempt")
        http_subcall: JsonValue | None = event.get("http_subcall")
        if (
            isinstance(logical_attempt, bool)
            or not isinstance(logical_attempt, int)
            or logical_attempt <= 0
            or isinstance(http_subcall, bool)
            or not isinstance(http_subcall, int)
            or http_subcall <= 0
        ):
            raise RuntimeMatchedGateError(
                "Usage attempt identifiers must be positive integers: "
                f"instance_id={instance_id}, logical_attempt={logical_attempt!r}, "
                f"http_subcall={http_subcall!r}"
            )
        event_id: tuple[str, int, int] = (
            instance_id,
            logical_attempt,
            http_subcall,
        )
        if event_id in event_ids:
            raise RuntimeMatchedGateError(
                f"Usage journal contains a duplicate event: event_id={event_id}"
            )
        event_ids.add(event_id)
        expected_fields: tuple[tuple[str, JsonValue], ...] = (
            ("schema_version", "runtime-matched-usage-event-v1"),
            ("job_id", job_id),
            ("model", served_model),
            ("domain", domain),
            ("arm", arm),
            ("answer_payload_hash", prepared_row["answer_payload_hash"]),
            ("execution_request_hash", prepared_row["execution_request_hash"]),
        )
        mismatches: list[str] = [
            f"{field}:expected={expected!r},observed={event.get(field)!r}"
            for field, expected in expected_fields
            if event.get(field) != expected
        ]
        if mismatches:
            raise RuntimeMatchedGateError(
                f"Usage event identity mismatch: event_id={event_id}, "
                f"mismatches={mismatches}"
            )
        usage_ids.add(instance_id)
    if usage_ids != set(output_index):
        raise RuntimeMatchedGateError(
            "Changed Gate output and usage support differ: "
            f"usage_without_output={sorted(usage_ids - set(output_index))[:20]}, "
            f"output_without_usage={sorted(set(output_index) - usage_ids)[:20]}"
        )


def base_answer_record(
    prepared: PreparedGateRequest,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    instances_sha256: str,
    corpus_sha256: str,
    model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    engine_attempts: int,
) -> JsonObject:
    """Build fields shared by changed Gate outcomes."""

    job: JsonObject = required_nested_object(
        manifest["runtime_facts"],
        "job",
        "runtime_facts",
    )
    instance_id: str = require_nonempty_string(
        prepared["instance"].get("instance_id"),
        "answer.instance_id",
    )
    expected_skill_ids: list[str] = prepared["expected_skill_ids"]
    return {
        "schema_version": GATE_RERUN_ANSWER_SCHEMA_VERSION,
        "stage": "answer",
        "job_id": job["job_id"],
        "instance_id": instance_id,
        "dataset": domain,
        "domain": domain,
        "arm": arm,
        "method": arm,
        "model": model,
        "served_model": served_model,
        "expected_skill_ids": expected_skill_ids,
        "actual_injection_state": {
            "state": "loaded" if expected_skill_ids else "bare",
            "skill_ids": expected_skill_ids,
        },
        "answer_payload_hash": prepared["answer_payload_hash"],
        "execution_request_hash": prepared["execution_request_hash"],
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "code_bundle_sha256": manifest["code_bundle_sha256"],
        "instances_sha256": instances_sha256,
        "corpus_sha256": corpus_sha256,
        "generation": manifest["generation"],
        "engine_attempts": engine_attempts,
        "reused_same_arm": False,
    }


def success_record(
    prepared: PreparedGateRequest,
    result: EngineResult,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    instances_sha256: str,
    corpus_sha256: str,
    model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    engine_attempts: int,
) -> JsonObject:
    """Build one successful changed Gate row."""

    record: JsonObject = base_answer_record(
        prepared,
        manifest,
        runtime_manifest_sha256,
        instances_sha256,
        corpus_sha256,
        model,
        served_model,
        domain,
        arm,
        engine_attempts,
    )
    record.update(
        {
            "raw_output": result.raw_output,
            "skill_ids_used": list(result.skill_ids_used),
            "failure_category": "success",
        }
    )
    if result.transcript is not None:
        record["transcript"] = result.transcript
    if result.meta:
        record["meta"] = cast(JsonObject, dict(result.meta))
    return record


def failure_record(
    prepared: PreparedGateRequest,
    category: FailureCategory,
    error_payload: Mapping[str, JsonLike],
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    instances_sha256: str,
    corpus_sha256: str,
    model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    engine_attempts: int,
) -> JsonObject:
    """Build one failed changed Gate row."""

    record: JsonObject = base_answer_record(
        prepared,
        manifest,
        runtime_manifest_sha256,
        instances_sha256,
        corpus_sha256,
        model,
        served_model,
        domain,
        arm,
        engine_attempts,
    )
    record.update(
        {
            "raw_output": "",
            "skill_ids_used": [],
            "failure_category": category,
            "error": cast(JsonObject, dict(error_payload)),
        }
    )
    return record


def run_one(
    prepared: PreparedGateRequest,
    engine: DirectEngine,
    client: object,
    manifest: JobBoundManifest,
    runtime_manifest_sha256: str,
    instances_sha256: str,
    corpus_sha256: str,
    model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    attempt_log_path: Path,
    request_error_types: tuple[type[Exception], ...],
    attempt_write_lock: threading.Lock,
    max_engine_attempts: int,
    retry_delays_seconds: tuple[float, ...],
) -> GateRunOutcome:
    """Run one changed Gate request with bounded retries."""

    if max_engine_attempts <= 0:
        raise ValueError(
            f"max_engine_attempts must be positive: value={max_engine_attempts}"
        )
    if len(retry_delays_seconds) != max_engine_attempts - 1:
        raise ValueError(
            "retry_delays_seconds must provide one delay per retry: "
            f"attempts={max_engine_attempts}, delays={retry_delays_seconds}"
        )
    instance_id: str = require_nonempty_string(
        prepared["instance"].get("instance_id"),
        "answer.instance_id",
    )
    job: JsonObject = required_nested_object(
        manifest["runtime_facts"],
        "job",
        "runtime_facts",
    )
    job_id: str = require_nonempty_string(job.get("job_id"), "job.job_id")
    for engine_attempt in range(1, max_engine_attempts + 1):
        context = ExecutionContext(
            job_id,
            served_model,
            domain,
            arm,
            instance_id,
            engine_attempt,
            prepared["answer_payload_hash"],
            prepared["execution_request_hash"],
        )
        started_at: float = time.monotonic()
        try:
            with bind_execution_context(context):
                result: EngineResult = engine.run(
                    prepared["instance"],
                    prepared["loaded_skills"],
                    client,
                    served_model,
                )
        except request_error_types as error:
            details = error_context(error)
            category: FailureCategory = classify_request_error(
                details.exception_name,
                details.message,
                details.status_code,
                details.response_body,
            )
            append_jsonl(
                attempt_log_path,
                attempt_payload(
                    context,
                    "error",
                    time.monotonic() - started_at,
                    {
                        "failure_category": category,
                        "exception_name": details.exception_name,
                        "message": details.message,
                        "status_code": details.status_code,
                        "response_body": details.response_body,
                    },
                ),
                attempt_write_lock,
            )
            if category == "infra_transient" and engine_attempt < max_engine_attempts:
                time.sleep(retry_delays_seconds[engine_attempt - 1])
                continue
            return {
                "record": failure_record(
                    prepared,
                    category,
                    {
                        "exception_name": details.exception_name,
                        "message": details.message,
                        "status_code": details.status_code,
                        "response_body": details.response_body,
                    },
                    manifest,
                    runtime_manifest_sha256,
                    instances_sha256,
                    corpus_sha256,
                    model,
                    served_model,
                    domain,
                    arm,
                    engine_attempt,
                ),
                "engine_attempts": engine_attempt,
            }
        expected_skill_ids: list[str] = prepared["expected_skill_ids"]
        if result.skill_ids_used != expected_skill_ids:
            raise RuntimeMatchedGateError(
                "Native direct engine reported unexpected Gate skills: "
                f"instance_id={instance_id}, expected={expected_skill_ids}, "
                f"observed={result.skill_ids_used}"
            )
        raw_output: object = result.raw_output
        if not isinstance(raw_output, str):
            raise RuntimeMatchedGateError(
                "Native direct engine returned non-string raw_output: "
                f"instance_id={instance_id}, "
                f"value_type={type(raw_output).__name__}"
            )
        append_jsonl(
            attempt_log_path,
            attempt_payload(
                context,
                "response",
                time.monotonic() - started_at,
                {"raw_output_empty": not bool(raw_output.strip())},
            ),
            attempt_write_lock,
        )
        if not raw_output.strip():
            if engine_attempt < max_engine_attempts:
                time.sleep(retry_delays_seconds[engine_attempt - 1])
                continue
            return {
                "record": failure_record(
                    prepared,
                    "method_failure",
                    {
                        "exception_name": "EmptyModelOutput",
                        "message": (
                            "Native direct engine returned empty output for all "
                            "bounded attempts"
                        ),
                        "status_code": None,
                        "response_body": "",
                    },
                    manifest,
                    runtime_manifest_sha256,
                    instances_sha256,
                    corpus_sha256,
                    model,
                    served_model,
                    domain,
                    arm,
                    engine_attempt,
                ),
                "engine_attempts": engine_attempt,
            }
        return {
            "record": success_record(
                prepared,
                result,
                manifest,
                runtime_manifest_sha256,
                instances_sha256,
                corpus_sha256,
                model,
                served_model,
                domain,
                arm,
                engine_attempt,
            ),
            "engine_attempts": engine_attempt,
        }
    raise AssertionError("Changed Gate retry loop exited without a terminal outcome")


def failure_category_counts(
    output_index: Mapping[str, JsonlLine],
) -> dict[str, int]:
    """Count changed Gate terminal categories."""

    counts: dict[str, int] = {}
    for line in output_index.values():
        category: FailureCategory = validate_failure_category(
            line["record"].get("failure_category")
        )
        counts[category] = counts.get(category, 0) + 1
    return counts


def run_gate_job(
    instances_path: Path,
    corpus_path: Path,
    gate_audit_path: Path,
    gate_decisions_path: Path,
    gate_rerun_path: Path,
    runtime_manifest_path: Path,
    repository_root: Path,
    output_path: Path,
    usage_log_path: Path,
    attempt_log_path: Path,
    api_base: str,
    workers: int,
    max_new_records: int,
    runtime: NativeBareRuntime,
    max_engine_attempts: int,
    retry_delays_seconds: tuple[float, ...],
) -> GateJobSummary:
    """Run a final-bound canary or full changed Gate resume."""

    if workers <= 0:
        raise ValueError(f"workers must be positive: value={workers}")
    if max_new_records < 0:
        raise ValueError(
            "max_new_records must be zero or positive: "
            f"value={max_new_records}"
        )
    resolved_paths: dict[str, Path] = {
        "instances": instances_path.resolve(),
        "corpus": corpus_path.resolve(),
        "gate_audit": gate_audit_path.resolve(),
        "gate_decisions": gate_decisions_path.resolve(),
        "gate_rerun": gate_rerun_path.resolve(),
    }
    resolved_manifest_path: Path = runtime_manifest_path.resolve()
    resolved_repository_root: Path = repository_root.resolve()
    resolved_output_path: Path = output_path.resolve()
    resolved_usage_path: Path = usage_log_path.resolve()
    resolved_attempt_path: Path = attempt_log_path.resolve()
    journal_paths: tuple[Path, ...] = (
        resolved_output_path,
        resolved_usage_path,
        resolved_attempt_path,
    )
    if len(set(journal_paths)) != len(journal_paths):
        raise RuntimeMatchedGateError(
            f"Output and journal paths must be distinct: paths={journal_paths}"
        )
    rerun_manifest: GateJsonObject = load_rerun_manifest(
        resolved_paths["gate_rerun"]
    )
    model: str = require_string(rerun_manifest.get("model"), "gate-rerun.model")
    served_model: str = require_string(
        rerun_manifest.get("served_model"),
        "gate-rerun.served_model",
    )
    domain: str = require_string(
        rerun_manifest.get("domain"),
        "gate-rerun.domain",
    )
    arm: GateArm = require_gate_arm(
        require_string(rerun_manifest.get("arm"), "gate-rerun.arm")
    )
    manifest: JobBoundManifest = load_job_bound_manifest(
        resolved_manifest_path
    )
    instances_sha256, corpus_sha256 = validate_gate_runtime_manifest(
        manifest,
        resolved_repository_root,
        resolved_paths,
        rerun_manifest,
        api_base,
        runtime,
    )
    runtime_manifest_sha256: str = sha256_file(resolved_manifest_path)
    instance_values: JsonValue = load_json_file(
        resolved_paths["instances"],
        "instances",
    )
    corpus_values: JsonValue = load_json_file(
        resolved_paths["corpus"],
        "corpus",
    )
    if not isinstance(instance_values, list) or not isinstance(corpus_values, list):
        raise RuntimeMatchedGateError(
            "Instances and corpus must both be JSON lists"
        )
    instances: dict[str, JsonObject] = cast(
        dict[str, JsonObject],
        index_instances(instance_values, domain),
    )
    corpus: dict[str, JsonObject] = cast(
        dict[str, JsonObject],
        index_corpus(corpus_values),
    )
    rerun_rows: dict[str, GateJsonObject] = index_rerun_rows(rerun_manifest)
    decisions: dict[str, GateJsonObject] = index_decisions(
        resolved_paths["gate_decisions"],
        model,
        served_model,
        domain,
        arm,
    )
    prepared: dict[str, PreparedGateRequest] = prepare_requests(
        rerun_rows,
        decisions,
        instances,
        corpus,
        served_model,
        runtime_manifest_sha256,
        manifest["code_bundle_sha256"],
        runtime,
    )
    output_lines: list[JsonlLine] = load_jsonl(
        resolved_output_path,
        "gate-answer-output",
        True,
    )
    output_index: dict[str, JsonlLine] = validate_existing_outputs(
        output_lines,
        prepared,
        model,
        served_model,
        domain,
        arm,
        runtime_manifest_sha256,
        manifest["code_bundle_sha256"],
    )
    job: JsonObject = required_nested_object(
        manifest["runtime_facts"],
        "job",
        "runtime_facts",
    )
    job_id: str = require_nonempty_string(job.get("job_id"), "job.job_id")
    usage_lines: list[JsonlLine] = load_jsonl(
        resolved_usage_path,
        "gate-usage",
        True,
    )
    validate_usage_journal(
        usage_lines,
        output_index,
        prepared,
        job_id,
        served_model,
        domain,
        arm,
    )
    pending_ids: list[str] = sorted(set(prepared) - set(output_index))
    selected_ids: list[str] = (
        pending_ids if max_new_records == 0 else pending_ids[:max_new_records]
    )
    output_lock: threading.Lock = threading.Lock()
    usage_lock: threading.Lock = threading.Lock()
    attempt_lock: threading.Lock = threading.Lock()
    completed_this_run: int = 0
    if selected_ids:
        raw_client: object = runtime["create_client"](api_base, None)
        client: OpenAIClientLike = wrap_openai_client(
            cast(OpenAIClientLike, raw_client),
            cast(UsageSink, JsonlUsageSink(resolved_usage_path, usage_lock)),
        )
        engine: DirectEngine = runtime["create_engine"](
            temperature=ANSWER_TEMPERATURE,
            max_tokens=ANSWER_MAX_TOKENS,
            thinking=ANSWER_THINKING,
        )
        futures: dict[Future[GateRunOutcome], str] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for instance_id in selected_ids:
                futures[
                    executor.submit(
                        run_one,
                        prepared[instance_id],
                        engine,
                        client,
                        manifest,
                        runtime_manifest_sha256,
                        instances_sha256,
                        corpus_sha256,
                        model,
                        served_model,
                        domain,
                        arm,
                        resolved_attempt_path,
                        runtime["request_error_types"],
                        attempt_lock,
                        max_engine_attempts,
                        retry_delays_seconds,
                    )
                ] = instance_id
            for future in as_completed(futures):
                outcome: GateRunOutcome = future.result()
                append_jsonl(
                    resolved_output_path,
                    outcome["record"],
                    output_lock,
                )
                completed_this_run += 1
                if (
                    completed_this_run % 100 == 0
                    or completed_this_run == len(futures)
                ):
                    print(
                        canonical_json(
                            {
                                "event": "runtime_matched_gate_progress",
                                "job_id": job_id,
                                "model": model,
                                "served_model": served_model,
                                "domain": domain,
                                "arm": arm,
                                "completed_this_run": completed_this_run,
                                "selected_this_run": len(selected_ids),
                            }
                        ),
                        flush=True,
                    )
    final_lines: list[JsonlLine] = load_jsonl(
        resolved_output_path,
        "gate-answer-output",
        True,
    )
    final_index: dict[str, JsonlLine] = validate_existing_outputs(
        final_lines,
        prepared,
        model,
        served_model,
        domain,
        arm,
        runtime_manifest_sha256,
        manifest["code_bundle_sha256"],
    )
    final_usage: list[JsonlLine] = load_jsonl(
        resolved_usage_path,
        "gate-usage",
        True,
    )
    validate_usage_journal(
        final_usage,
        final_index,
        prepared,
        job_id,
        served_model,
        domain,
        arm,
    )
    categories: dict[str, int] = failure_category_counts(final_index)
    unresolved: int = (
        categories.get("infra_transient", 0)
        + categories.get("unclassified_error", 0)
    )
    missing_after_run: int = len(set(prepared) - set(final_index))
    full_run: bool = max_new_records == 0
    run_valid: bool = (
        completed_this_run == len(selected_ids)
        and unresolved == 0
        and (not full_run or missing_after_run == 0)
    )
    return {
        "event": "runtime_matched_gate_job_state",
        "job_id": job_id,
        "model": model,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "run_mode": "full" if full_run else "canary",
        "expected_changed_rows": len(prepared),
        "observed_changed_rows": len(final_index),
        "pending_before_run": len(pending_ids),
        "selected_this_run": len(selected_ids),
        "completed_this_run": completed_this_run,
        "unresolved": unresolved,
        "missing_after_run": missing_after_run,
        "failure_categories": categories,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "output": str(resolved_output_path),
        "output_sha256": (
            sha256_file(resolved_output_path)
            if resolved_output_path.is_file()
            else ""
        ),
        "run_valid": run_valid,
    }


def main() -> None:
    """Execute one changed Gate canary or full resume."""

    args = parse_args()
    summary: GateJobSummary = run_gate_job(
        cast(Path, args.instances),
        cast(Path, args.corpus),
        cast(Path, args.gate_audit),
        cast(Path, args.gate_decisions),
        cast(Path, args.gate_rerun),
        cast(Path, args.runtime_manifest),
        cast(Path, args.repository_root),
        cast(Path, args.output),
        cast(Path, args.usage_log),
        cast(Path, args.attempt_log),
        str(args.api_base),
        int(args.workers),
        int(args.max_new_records),
        load_native_bare_runtime(),
        MAX_ENGINE_ATTEMPTS,
        RETRY_DELAYS_SECONDS,
    )
    print(canonical_json(summary), flush=True)
    if not summary["run_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
