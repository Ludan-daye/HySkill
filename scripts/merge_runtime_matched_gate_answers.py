#!/usr/bin/env python3
"""Merge changed Gate reruns while preserving unchanged K=2 rows exactly."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from hyskill.runtime_matched_execution import (
    FileEvidence,
    JobBoundManifest,
    JsonValue,
    canonical_json,
    execution_request_hash,
    load_job_bound_manifest,
    require_sha256,
    sha256_bytes,
    sha256_file,
    sha256_json,
    validate_frozen_k2_runtime_reference,
    verify_job_bound_manifest_files,
)
from hyskill.runtime_matched_gate import (
    ANSWER_MAX_TOKENS,
    ANSWER_PAYLOAD_SCHEMA_VERSION,
    ANSWER_TEMPERATURE,
    ANSWER_THINKING,
    GATE_MERGE_REPORT_SCHEMA_VERSION,
    GATE_RERUN_ANSWER_SCHEMA_VERSION,
    GateArm,
    GateTaskKey,
    JsonObject,
    RuntimeMatchedGateError,
    expected_gate_task_keys,
    require_boolean,
    require_object,
    require_string,
    require_string_list,
)
from scripts.summarize_runtime_matched_gate import (
    GateTaskArtifacts,
    load_gate_task,
)


class RawJsonlLine(TypedDict):
    """One parsed JSONL record and its unchanged source bytes."""

    line_number: int
    raw_bytes: bytes
    record: JsonObject


class GateMergeResult(TypedDict):
    """Pure merge output and recomputed validation counts."""

    output_lines: list[bytes]
    expected_count: int
    preserved_count: int
    rerun_count: int
    preserved_method_failure_count: int
    rerun_success_count: int
    rerun_method_failure_count: int
    preserved_line_bundle_sha256: str
    rerun_line_bundle_sha256: str


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-domain-arm Gate answer merge."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--old-answers", required=True, type=Path)
    parser.add_argument("--rerun-answers", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument(
        "--artifact-path",
        action="append",
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    return parser.parse_args()


def read_jsonl_bytes(
    path: Path,
    context: str,
    allow_empty: bool,
) -> list[RawJsonlLine]:
    """Read strict newline-terminated JSONL without normalizing any bytes."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    raw_file: bytes = path.read_bytes()
    if not raw_file:
        if allow_empty:
            return []
        raise RuntimeMatchedGateError(f"{context} is empty: path={path}")
    if not raw_file.endswith(b"\n"):
        raise RuntimeMatchedGateError(
            f"{context} must end with a newline for exact row preservation: "
            f"path={path}"
        )
    output: list[RawJsonlLine] = []
    for line_number, raw_line in enumerate(
        raw_file.splitlines(keepends=True),
        start=1,
    ):
        if not raw_line.strip():
            raise RuntimeMatchedGateError(
                f"{context} contains a blank line: "
                f"path={path}, line={line_number}"
            )
        try:
            text_line: str = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeMatchedGateError(
                f"{context} is not valid UTF-8: "
                f"path={path}, line={line_number}, offset={error.start}"
            ) from error
        try:
            raw_value: JsonValue = cast(JsonValue, json.loads(text_line))
        except json.JSONDecodeError as error:
            raise RuntimeMatchedGateError(
                f"{context} JSONL is malformed: path={path}, "
                f"line={line_number}, column={error.colno}, "
                f"message={error.msg}"
            ) from error
        output.append(
            {
                "line_number": line_number,
                "raw_bytes": raw_line,
                "record": require_object(
                    raw_value,
                    f"{context}:{path}:{line_number}",
                ),
            }
        )
    return output


def index_raw_lines(
    lines: Sequence[RawJsonlLine],
    context: str,
) -> dict[str, RawJsonlLine]:
    """Index raw JSONL lines by one required unique instance ID."""

    output: dict[str, RawJsonlLine] = {}
    for line in lines:
        instance_id: str = require_string(
            line["record"].get("instance_id"),
            f"{context}:{line['line_number']}.instance_id",
        )
        if instance_id in output:
            raise RuntimeMatchedGateError(
                f"{context} contains duplicate instance: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = line
    return output


def _task_identity(
    task: GateTaskArtifacts,
) -> tuple[str, str, str, GateArm]:
    """Return one validated task identity in answer-row field order."""

    key: GateTaskKey = task["key"]
    if key not in set(expected_gate_task_keys()):
        raise RuntimeMatchedGateError(
            f"Gate merge uses an unsupported task: key={key}"
        )
    return (
        task["model"],
        task["served_model"],
        task["domain"],
        task["arm"],
    )


def validate_runtime_manifest(
    path: Path,
    repository_root: Path,
    task: GateTaskArtifacts,
    artifact_paths: Mapping[str, Path],
) -> tuple[JobBoundManifest, str]:
    """Validate the changed-row job runtime and return its file SHA."""

    manifest: JobBoundManifest = load_job_bound_manifest(path)
    verification_manifest: JobBoundManifest = (
        manifest_with_artifact_path_mapping(manifest, artifact_paths)
    )
    verify_job_bound_manifest_files(verification_manifest, repository_root)
    validate_frozen_k2_runtime_reference(manifest["runtime_facts"])
    model, served_model, domain, arm = _task_identity(task)
    runtime_facts: JsonObject = manifest["runtime_facts"]
    job: JsonObject = require_object(
        runtime_facts.get("job"),
        "runtime_manifest.runtime_facts.job",
    )
    endpoint: JsonObject = require_object(
        runtime_facts.get("endpoint"),
        "runtime_manifest.runtime_facts.endpoint",
    )
    expected_job_fields: tuple[tuple[str, JsonValue], ...] = (
        ("result_tag", model),
        ("model", served_model),
        ("domain", domain),
        ("arm", arm),
    )
    expected_endpoint_fields: tuple[tuple[str, JsonValue], ...] = (
        ("served_model", served_model),
        ("dtype", "bfloat16"),
        ("quantization", "none"),
        ("max_model_len", 8192),
    )
    mismatches: list[str] = [
        f"job.{field}:expected={expected!r},observed={job.get(field)!r}"
        for field, expected in expected_job_fields
        if job.get(field) != expected
    ]
    mismatches.extend(
        f"endpoint.{field}:expected={expected!r},"
        f"observed={endpoint.get(field)!r}"
        for field, expected in expected_endpoint_fields
        if endpoint.get(field) != expected
    )
    generation: JsonObject = manifest["generation"]
    expected_generation: tuple[tuple[str, JsonValue], ...] = (
        ("temperature", ANSWER_TEMPERATURE),
        ("max_tokens", ANSWER_MAX_TOKENS),
        ("thinking", ANSWER_THINKING),
    )
    mismatches.extend(
        f"generation.{field}:expected={expected!r},"
        f"observed={generation.get(field)!r}"
        for field, expected in expected_generation
        if generation.get(field) != expected
    )
    allowed_generation_fields: set[str] = {
        "temperature",
        "max_tokens",
        "thinking",
        "extra_body",
    }
    unexpected_generation_fields: list[str] = sorted(
        set(generation) - allowed_generation_fields
    )
    if unexpected_generation_fields:
        mismatches.append(
            "generation.unexpected_fields:"
            f"observed={unexpected_generation_fields}"
        )
    if "extra_body" not in generation:
        mismatches.append("generation.extra_body:missing")
    if mismatches:
        raise RuntimeMatchedGateError(
            "Changed Gate runtime manifest identity mismatch: "
            f"key={task['key']}, mismatches={mismatches}"
        )
    return manifest, sha256_file(path)


def parse_artifact_path_mappings(
    values: Sequence[str],
) -> dict[str, Path]:
    """Parse explicit unique artifact-name path mappings."""

    output: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise RuntimeMatchedGateError(
                "Artifact path mapping must use NAME=PATH: "
                f"value={value!r}"
            )
        if name in output:
            raise RuntimeMatchedGateError(
                f"Artifact path mapping is duplicated: name={name}"
            )
        output[name] = Path(raw_path).resolve()
    return output


def manifest_with_artifact_path_mapping(
    manifest: JobBoundManifest,
    artifact_paths: Mapping[str, Path],
) -> JobBoundManifest:
    """Bind unchanged artifact evidence to explicit local file locations."""

    if not artifact_paths:
        return manifest
    declared_names: set[str] = {
        artifact["name"] for artifact in manifest["artifacts"]
    }
    mapped_names: set[str] = set(artifact_paths)
    if mapped_names != declared_names:
        raise RuntimeMatchedGateError(
            "Artifact path mappings must exactly cover the runtime manifest: "
            f"missing={sorted(declared_names - mapped_names)}, "
            f"unexpected={sorted(mapped_names - declared_names)}"
        )
    mapped_artifacts: list[FileEvidence] = [
        {
            "name": artifact["name"],
            "path": str(artifact_paths[artifact["name"]]),
            "size_bytes": artifact["size_bytes"],
            "sha256": artifact["sha256"],
        }
        for artifact in manifest["artifacts"]
    ]
    return {
        "schema_version": manifest["schema_version"],
        "runtime_facts": manifest["runtime_facts"],
        "generation": manifest["generation"],
        "artifacts": mapped_artifacts,
        "code_files": manifest["code_files"],
        "code_bundle_sha256": manifest["code_bundle_sha256"],
    }


def expected_preserved_answer_model(
    result_tag: str,
    served_model: str,
) -> str:
    """Return the frozen legacy answer-row model tag."""

    if (
        result_tag == "qwen3.5-4b-reference"
        and served_model == "qwen3.5-4b"
    ):
        return served_model
    return result_tag


def validate_old_answer(
    row: Mapping[str, JsonValue],
    diff: Mapping[str, JsonValue],
    model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
) -> str:
    """Validate one original K=2 row against the old side of its diff."""

    instance_id: str = require_string(
        diff.get("instance_id"),
        "diff.instance_id",
    )
    preserved_model: str = expected_preserved_answer_model(
        model,
        served_model,
    )
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("schema_version", "k2-answer-record-v1"),
        ("instance_id", instance_id),
        ("model", preserved_model),
        ("served_model", served_model),
        ("dataset", domain),
        ("method", arm),
        ("expected_skill_ids", diff["old_expected_skill_ids"]),
        ("failure_category", diff["old_failure_category"]),
        ("request_hash", diff["old_request_hash"]),
    )
    mismatches: list[str] = [
        f"{field}:expected={expected!r},observed={row.get(field)!r}"
        for field, expected in expected_fields
        if row.get(field) != expected
    ]
    if mismatches:
        raise RuntimeMatchedGateError(
            "Original K=2 Gate answer differs from the audited old row: "
            f"instance_id={instance_id}, mismatches={mismatches}"
        )
    failure_category: str = require_string(
        row.get("failure_category"),
        f"old-answer:{instance_id}.failure_category",
    )
    raw_output: JsonValue | None = row.get("raw_output")
    if not isinstance(raw_output, str):
        raise RuntimeMatchedGateError(
            f"Original K=2 raw_output is not a string: "
            f"instance_id={instance_id}"
        )
    expected_skills: list[str] = require_string_list(
        row.get("expected_skill_ids"),
        f"old-answer:{instance_id}.expected_skill_ids",
    )
    skill_ids_used: list[str] = require_string_list(
        row.get("skill_ids_used"),
        f"old-answer:{instance_id}.skill_ids_used",
    )
    injection_state: JsonObject = require_object(
        row.get("actual_injection_state"),
        f"old-answer:{instance_id}.actual_injection_state",
    )
    submitted_skills: list[str] = require_string_list(
        injection_state.get("skill_ids"),
        f"old-answer:{instance_id}.actual_injection_state.skill_ids",
    )
    if submitted_skills != expected_skills:
        raise RuntimeMatchedGateError(
            "Original K=2 submitted skills differ from its decision: "
            f"instance_id={instance_id}, expected={expected_skills}, "
            f"observed={submitted_skills}"
        )
    if failure_category == "success":
        if not raw_output.strip() or skill_ids_used != expected_skills:
            raise RuntimeMatchedGateError(
                "Original successful K=2 row has invalid output or skills: "
                f"instance_id={instance_id}"
            )
    elif failure_category == "method_failure":
        if raw_output or skill_ids_used:
            raise RuntimeMatchedGateError(
                "Original K=2 method failure carries output or used skills: "
                f"instance_id={instance_id}"
            )
    else:
        raise RuntimeMatchedGateError(
            "Original K=2 row has an unresolved outcome: "
            f"instance_id={instance_id}, category={failure_category}"
        )
    runtime_identity: JsonObject = require_object(
        row.get("runtime_identity"),
        f"old-answer:{instance_id}.runtime_identity",
    )
    if (
        runtime_identity.get("model") != model
        or runtime_identity.get("served_model") != served_model
    ):
        raise RuntimeMatchedGateError(
            "Original K=2 runtime identity mismatch: "
            f"instance_id={instance_id}, runtime_identity={runtime_identity}"
        )
    return failure_category


def validate_rerun_answer(
    row: Mapping[str, JsonValue],
    diff: Mapping[str, JsonValue],
    model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
) -> str:
    """Validate one fresh changed-payload answer and its execution identity."""

    instance_id: str = require_string(
        diff.get("instance_id"),
        "diff.instance_id",
    )
    payload_hash: str = require_sha256(
        diff.get("new_answer_payload_hash"),
        f"diff:{instance_id}.new_answer_payload_hash",
    )
    expected_execution_hash: str = execution_request_hash(
        ANSWER_PAYLOAD_SCHEMA_VERSION,
        payload_hash,
        runtime_manifest_sha256,
        code_bundle_sha256,
    )
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("schema_version", GATE_RERUN_ANSWER_SCHEMA_VERSION),
        ("instance_id", instance_id),
        ("model", model),
        ("served_model", served_model),
        ("domain", domain),
        ("arm", arm),
        ("stage", "answer"),
        ("expected_skill_ids", diff["new_expected_skill_ids"]),
        ("answer_payload_hash", payload_hash),
        ("execution_request_hash", expected_execution_hash),
        ("runtime_manifest_sha256", runtime_manifest_sha256),
        ("code_bundle_sha256", code_bundle_sha256),
        ("reused_same_arm", False),
    )
    mismatches: list[str] = [
        f"{field}:expected={expected!r},observed={row.get(field)!r}"
        for field, expected in expected_fields
        if row.get(field) != expected
    ]
    if mismatches:
        raise RuntimeMatchedGateError(
            "Changed Gate answer identity mismatch: "
            f"instance_id={instance_id}, mismatches={mismatches}"
        )
    require_sha256(
        row.get("execution_request_hash"),
        f"rerun-answer:{instance_id}.execution_request_hash",
    )
    raw_output: JsonValue | None = row.get("raw_output")
    if not isinstance(raw_output, str):
        raise RuntimeMatchedGateError(
            f"Changed Gate raw_output is not a string: "
            f"instance_id={instance_id}"
        )
    expected_skills: list[str] = require_string_list(
        row.get("expected_skill_ids"),
        f"rerun-answer:{instance_id}.expected_skill_ids",
    )
    skill_ids_used: list[str] = require_string_list(
        row.get("skill_ids_used"),
        f"rerun-answer:{instance_id}.skill_ids_used",
    )
    injection_state: JsonObject = require_object(
        row.get("actual_injection_state"),
        f"rerun-answer:{instance_id}.actual_injection_state",
    )
    submitted_skills: list[str] = require_string_list(
        injection_state.get("skill_ids"),
        f"rerun-answer:{instance_id}.actual_injection_state.skill_ids",
    )
    if submitted_skills != expected_skills:
        raise RuntimeMatchedGateError(
            "Changed Gate request submitted unexpected skills: "
            f"instance_id={instance_id}, expected={expected_skills}, "
            f"observed={submitted_skills}"
        )
    failure_category: str = require_string(
        row.get("failure_category"),
        f"rerun-answer:{instance_id}.failure_category",
    )
    if failure_category == "success":
        if not raw_output.strip() or skill_ids_used != expected_skills:
            raise RuntimeMatchedGateError(
                "Changed Gate success has invalid output or used skills: "
                f"instance_id={instance_id}"
            )
    elif failure_category == "method_failure":
        if raw_output or skill_ids_used:
            raise RuntimeMatchedGateError(
                "Changed Gate method failure carries output or used skills: "
                f"instance_id={instance_id}"
            )
    else:
        raise RuntimeMatchedGateError(
            "Changed Gate rerun has an unresolved outcome: "
            f"instance_id={instance_id}, category={failure_category}"
        )
    return failure_category


def _line_bundle_sha256(
    lines: Sequence[tuple[str, bytes]],
) -> str:
    evidence: list[JsonObject] = [
        {
            "instance_id": instance_id,
            "line_sha256": sha256_bytes(raw_line),
        }
        for instance_id, raw_line in lines
    ]
    return sha256_json(evidence)


def merge_gate_answer_lines(
    old_lines: Sequence[RawJsonlLine],
    rerun_lines: Sequence[RawJsonlLine],
    diff_rows: Sequence[JsonObject],
    model: str,
    served_model: str,
    domain: str,
    arm: GateArm,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
) -> GateMergeResult:
    """Merge one task using old order and exact selected source line bytes."""

    require_sha256(runtime_manifest_sha256, "runtime_manifest_sha256")
    require_sha256(code_bundle_sha256, "code_bundle_sha256")
    old_by_id: dict[str, RawJsonlLine] = index_raw_lines(
        old_lines,
        "old answers",
    )
    rerun_by_id: dict[str, RawJsonlLine] = index_raw_lines(
        rerun_lines,
        "rerun answers",
    )
    diff_by_id: dict[str, JsonObject] = {}
    for index, diff_row in enumerate(diff_rows):
        diff_instance_id: str = require_string(
            diff_row.get("instance_id"),
            f"diff[{index}].instance_id",
        )
        if diff_instance_id in diff_by_id:
            raise RuntimeMatchedGateError(
                f"Gate diff contains duplicate instance: "
                f"instance_id={diff_instance_id}"
            )
        diff_by_id[diff_instance_id] = diff_row
    if set(old_by_id) != set(diff_by_id):
        raise RuntimeMatchedGateError(
            "Original K=2 answers do not exactly cover the Gate diff: "
            f"missing={sorted(set(diff_by_id) - set(old_by_id))[:20]}, "
            f"unexpected={sorted(set(old_by_id) - set(diff_by_id))[:20]}"
        )
    changed_ids: set[str] = {
        instance_id
        for instance_id, diff in diff_by_id.items()
        if require_boolean(
            diff.get("rerun_required"),
            f"diff:{instance_id}.rerun_required",
        )
    }
    if set(rerun_by_id) != changed_ids:
        raise RuntimeMatchedGateError(
            "Changed Gate answers must contain exactly the rerun-required IDs: "
            f"missing={sorted(changed_ids - set(rerun_by_id))[:20]}, "
            f"unexpected={sorted(set(rerun_by_id) - changed_ids)[:20]}"
        )
    output_lines: list[bytes] = []
    preserved_lines: list[tuple[str, bytes]] = []
    replaced_lines: list[tuple[str, bytes]] = []
    preserved_method_failures: int = 0
    rerun_successes: int = 0
    rerun_method_failures: int = 0
    for old_line in old_lines:
        current_instance_id: str = require_string(
            old_line["record"].get("instance_id"),
            f"old-answer:{old_line['line_number']}.instance_id",
        )
        current_diff: JsonObject = diff_by_id[current_instance_id]
        old_category: str = validate_old_answer(
            old_line["record"],
            current_diff,
            model,
            served_model,
            domain,
            arm,
        )
        if current_instance_id not in changed_ids:
            selected_bytes: bytes = old_line["raw_bytes"]
            if selected_bytes != old_line["raw_bytes"]:
                raise AssertionError(
                    f"Unchanged Gate row bytes were altered: "
                    f"instance_id={current_instance_id}"
                )
            output_lines.append(selected_bytes)
            preserved_lines.append((current_instance_id, selected_bytes))
            if old_category == "method_failure":
                preserved_method_failures += 1
            continue
        rerun_line: RawJsonlLine = rerun_by_id[current_instance_id]
        rerun_category: str = validate_rerun_answer(
            rerun_line["record"],
            current_diff,
            model,
            served_model,
            domain,
            arm,
            runtime_manifest_sha256,
            code_bundle_sha256,
        )
        output_lines.append(rerun_line["raw_bytes"])
        replaced_lines.append(
            (current_instance_id, rerun_line["raw_bytes"])
        )
        if rerun_category == "success":
            rerun_successes += 1
        else:
            rerun_method_failures += 1
    if len(output_lines) != len(diff_rows):
        raise AssertionError(
            "Gate merge output count changed after exact coverage validation"
        )
    return {
        "output_lines": output_lines,
        "expected_count": len(diff_rows),
        "preserved_count": len(preserved_lines),
        "rerun_count": len(replaced_lines),
        "preserved_method_failure_count": preserved_method_failures,
        "rerun_success_count": rerun_successes,
        "rerun_method_failure_count": rerun_method_failures,
        "preserved_line_bundle_sha256": _line_bundle_sha256(
            preserved_lines
        ),
        "rerun_line_bundle_sha256": _line_bundle_sha256(replaced_lines),
    }


def write_bytes_atomic(path: Path, chunks: Sequence[bytes]) -> None:
    """Atomically write a sequence of already validated source lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("wb") as output_file:
        for chunk in chunks:
            output_file.write(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary_path.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, JsonValue]) -> None:
    """Atomically write one formatted JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def declared_input_sha256(
    task: GateTaskArtifacts,
    artifact_name: str,
) -> str:
    """Return one SHA declared by the task audit input inventory."""

    input_artifacts: JsonObject = require_object(
        task["audit"].get("input_artifacts"),
        "audit.input_artifacts",
    )
    artifact: JsonObject = require_object(
        input_artifacts.get(artifact_name),
        f"audit.input_artifacts.{artifact_name}",
    )
    return require_sha256(
        artifact.get("sha256"),
        f"audit.input_artifacts.{artifact_name}.sha256",
    )


def main() -> None:
    """Validate and merge one changed-row-only Gate answer task."""

    args = parse_args()
    audit_path: Path = cast(Path, args.audit).resolve()
    old_answers_path: Path = cast(Path, args.old_answers).resolve()
    rerun_answers_path: Path = cast(Path, args.rerun_answers).resolve()
    runtime_manifest_path: Path = cast(
        Path,
        args.runtime_manifest,
    ).resolve()
    repository_root: Path = cast(Path, args.repository_root).resolve()
    raw_artifact_paths: list[str] | None = cast(
        list[str] | None,
        args.artifact_path,
    )
    artifact_paths: dict[str, Path] = parse_artifact_path_mappings(
        [] if raw_artifact_paths is None else raw_artifact_paths
    )
    output_path: Path = cast(Path, args.output).resolve()
    report_output_path: Path = cast(Path, args.report_output).resolve()
    input_paths: set[Path] = {
        audit_path,
        old_answers_path,
        rerun_answers_path,
        runtime_manifest_path,
        *artifact_paths.values(),
    }
    if output_path == report_output_path:
        raise RuntimeMatchedGateError(
            f"Gate merge outputs must be distinct: path={output_path}"
        )
    output_aliases: set[Path] = {
        path
        for path in (output_path, report_output_path)
        if path in input_paths
    }
    if output_aliases:
        raise RuntimeMatchedGateError(
            "Gate merge must not overwrite an input artifact: "
            f"aliases={sorted(output_aliases)}"
        )
    hash_cache: dict[Path, str] = {}
    task: GateTaskArtifacts = load_gate_task(audit_path, hash_cache)
    verified_source_aliases: set[Path] = {
        path
        for path in (output_path, report_output_path)
        if path in hash_cache
    }
    if verified_source_aliases:
        raise RuntimeMatchedGateError(
            "Gate merge outputs must not overwrite any audit-bound source "
            f"artifact: aliases={sorted(verified_source_aliases)}"
        )
    model, served_model, domain, arm = _task_identity(task)
    observed_old_sha256: str = sha256_file(old_answers_path)
    expected_old_sha256: str = declared_input_sha256(task, "old_answers")
    if observed_old_sha256 != expected_old_sha256:
        raise RuntimeMatchedGateError(
            "Gate merge old-answer SHA differs from the audited source: "
            f"expected={expected_old_sha256}, observed={observed_old_sha256}, "
            f"path={old_answers_path}"
        )
    manifest, runtime_manifest_sha256 = validate_runtime_manifest(
        runtime_manifest_path,
        repository_root,
        task,
        artifact_paths,
    )
    old_lines: list[RawJsonlLine] = read_jsonl_bytes(
        old_answers_path,
        "old answers",
        False,
    )
    rerun_lines: list[RawJsonlLine] = read_jsonl_bytes(
        rerun_answers_path,
        "rerun answers",
        True,
    )
    result: GateMergeResult = merge_gate_answer_lines(
        old_lines,
        rerun_lines,
        task["diff_rows"],
        model,
        served_model,
        domain,
        arm,
        runtime_manifest_sha256,
        manifest["code_bundle_sha256"],
    )
    if result["expected_count"] != task["expected_count"]:
        raise RuntimeMatchedGateError(
            "Gate merge result count differs from the audited denominator: "
            f"expected={task['expected_count']}, "
            f"observed={result['expected_count']}"
        )
    write_bytes_atomic(output_path, result["output_lines"])
    expected_output_bytes: bytes = b"".join(result["output_lines"])
    observed_output_bytes: bytes = output_path.read_bytes()
    if observed_output_bytes != expected_output_bytes:
        raise RuntimeMatchedGateError(
            "Gate merge output differs from the selected exact source lines: "
            f"path={output_path}"
        )
    output_lines: list[RawJsonlLine] = read_jsonl_bytes(
        output_path,
        "merged answers",
        False,
    )
    if len(output_lines) != task["expected_count"]:
        raise RuntimeMatchedGateError(
            "Gate merge output failed read-back count validation: "
            f"expected={task['expected_count']}, observed={len(output_lines)}"
        )
    rerun_answers_sha256: str = sha256_file(rerun_answers_path)
    report: JsonObject = {
        "schema_version": GATE_MERGE_REPORT_SCHEMA_VERSION,
        "valid": True,
        "model": model,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "expected_count": task["expected_count"],
        "observed_count": len(output_lines),
        "preserved_row_count": result["preserved_count"],
        "rerun_row_count": result["rerun_count"],
        "preserved_method_failure_count": (
            result["preserved_method_failure_count"]
        ),
        "rerun_success_count": result["rerun_success_count"],
        "rerun_method_failure_count": result[
            "rerun_method_failure_count"
        ],
        "unresolved_outcome_count": 0,
        "unchanged_rows_preserved_byte_for_byte": True,
        "preserved_line_bundle_sha256": result[
            "preserved_line_bundle_sha256"
        ],
        "rerun_line_bundle_sha256": result["rerun_line_bundle_sha256"],
        "audit": {
            "path": str(task["audit_path"]),
            "sha256": task["audit_sha256"],
        },
        "diff": {
            "path": str(task["diff_path"]),
            "sha256": task["diff_sha256"],
        },
        "old_answers": {
            "path": str(old_answers_path),
            "sha256": observed_old_sha256,
        },
        "rerun_answers": {
            "path": str(rerun_answers_path),
            "sha256": rerun_answers_sha256,
        },
        "runtime_manifest": {
            "path": str(runtime_manifest_path),
            "sha256": runtime_manifest_sha256,
            "code_bundle_sha256": manifest["code_bundle_sha256"],
        },
        "merged_answers": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        },
    }
    report["evidence_bundle_sha256"] = sha256_json(report)
    write_json_atomic(report_output_path, report)
    print(
        canonical_json(
            {
                "event": "runtime_matched_gate_merge_complete",
                "model": model,
                "domain": domain,
                "arm": arm,
                "preserved": result["preserved_count"],
                "rerun": result["rerun_count"],
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "report": str(report_output_path),
                "report_sha256": sha256_file(report_output_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
