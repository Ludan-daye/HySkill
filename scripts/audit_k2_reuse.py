#!/usr/bin/env python3
"""Audit strict same-arm K=4 answer reuse for one K=2 answer job."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict, cast

from hyskill.downstream_reuse import (
    AnswerGeneration,
    DownstreamDataError,
    FailureCategory,
    JsonLike,
    JsonObject,
    JsonValue,
    PreseedEligibility,
    RuntimeManifest,
    SemanticArm,
    allowed_legacy_label,
    always_expected_skill_ids,
    answer_execution_fingerprint,
    audit_record_coverage,
    canonical_json,
    derive_legacy_answer_success,
    gated_expected_skill_ids,
    normalize_skill_ids,
    require_runtime_code_files,
    same_arm_preseed_eligibility,
    sha256_file,
    sha256_text,
    validate_answer_runtime_manifest,
    validate_legacy_manifest_evidence,
)


ANSWER_SCHEMA_VERSION: str = "k2-answer-request-v1"
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 2048
THINKING: bool = False


class BuildPrompt(Protocol):
    """SR-Agents dataset prompt builder contract."""

    def __call__(
        self,
        instance: JsonObject,
        skills: list[str] | None,
    ) -> tuple[str, str]:
        """Build exact answer messages."""


class GetExtraBody(Protocol):
    """SR-Agents thinking-control helper contract."""

    def __call__(
        self,
        model: str,
        thinking: bool,
    ) -> JsonObject | None:
        """Return request body additions."""


class AnswerRuntime(TypedDict):
    """Exact prompt and model-config functions from SR-Agents."""

    build_prompt: BuildPrompt
    get_extra_body: GetExtraBody


class LegacyLine(TypedDict):
    """One source answer row with byte-level provenance."""

    line_number: int
    raw_line: str
    line_sha256: str
    record: JsonObject


class AuditRecord(TypedDict):
    """One per-instance strict-reuse decision."""

    instance_id: str
    arm: SemanticArm
    legacy_label: str | None
    status: str
    reason: str
    needs_inference: bool
    expected_skill_ids: list[str]
    legacy_skill_ids: list[str]
    new_request_hash: str
    old_request_hash: str | None
    source_jsonl_sha256: str
    source_line_number: int | None
    source_line_sha256: str | None
    runtime_identity_matches: bool


def parse_args() -> argparse.Namespace:
    """Parse explicit input, identity, and output paths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--decision-source", required=True, type=Path)
    parser.add_argument("--legacy-jsonl", required=True, type=Path)
    parser.add_argument("--old-runtime-manifest", required=True, type=Path)
    parser.add_argument("--new-runtime-manifest", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=(
            "routed_always",
            "routed_gated",
            "routed_select",
            "fixed_gated",
        ),
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--preseed-output", required=True, type=Path)
    parser.add_argument("--pending-output", required=True, type=Path)
    return parser.parse_args()


def load_answer_runtime() -> AnswerRuntime:
    """Load exact request rendering from the installed SR-Agents."""

    try:
        prompts_module: ModuleType = importlib.import_module("sragents.prompts")
        llm_module: ModuleType = importlib.import_module("sragents.llm")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SR-Agents runtime is unavailable. Install the project environment "
            "before auditing answer reuse: "
            "command=.venv/bin/pip install --no-deps -e external/SR-Agents; "
            "install any missing packages separately from an approved China mirror"
        ) from error
    return {
        "build_prompt": cast(BuildPrompt, getattr(prompts_module, "build_prompt")),
        "get_extra_body": cast(
            GetExtraBody, getattr(llm_module, "get_extra_body")
        ),
    }


def load_json(path: Path) -> JsonValue:
    """Load one UTF-8 JSON file with path-aware errors."""

    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: path={path}")
    try:
        return cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise DownstreamDataError(
            "JSON file is malformed: "
            f"path={path}, line={error.lineno}, column={error.colno}, "
            f"message={error.msg}"
        ) from error


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return a JSON object or raise with context."""

    if not isinstance(value, dict):
        raise DownstreamDataError(
            f"Expected JSON object: context={context}, actual={type(value).__name__}"
        )
    return cast(JsonObject, value)


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return a JSON list or raise with context."""

    if not isinstance(value, list):
        raise DownstreamDataError(
            f"Expected JSON list: context={context}, actual={type(value).__name__}"
        )
    return cast(list[JsonValue], value)


def require_string(value: JsonValue | None, context: str) -> str:
    """Return a non-empty string or raise with context."""

    if not isinstance(value, str) or not value:
        raise DownstreamDataError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def load_instances(path: Path, domain: str) -> list[JsonObject]:
    """Load unique instances from exactly one domain."""

    values: list[JsonValue] = require_list(load_json(path), f"instances:{path}")
    rows: list[JsonObject] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(values):
        row: JsonObject = require_object(
            value,
            f"instances:{path}[{index}]",
        )
        instance_id: str = require_string(
            row.get("instance_id"),
            f"instances:{path}[{index}].instance_id",
        )
        if instance_id in seen_ids:
            raise DownstreamDataError(
                f"Duplicate instance ID: path={path}, instance_id={instance_id}"
            )
        if row.get("dataset") != domain:
            raise DownstreamDataError(
                "Instance domain mismatch: "
                f"path={path}, instance_id={instance_id}, "
                f"expected={domain}, actual={row.get('dataset')!r}"
            )
        seen_ids.add(instance_id)
        rows.append(row)
    return rows


def load_corpus(path: Path) -> dict[str, JsonObject]:
    """Load a unique skill corpus indexed by skill_id."""

    values: list[JsonValue] = require_list(load_json(path), f"corpus:{path}")
    output: dict[str, JsonObject] = {}
    for index, value in enumerate(values):
        skill: JsonObject = require_object(
            value,
            f"corpus:{path}[{index}]",
        )
        skill_id: str = require_string(
            skill.get("skill_id"),
            f"corpus:{path}[{index}].skill_id",
        )
        if skill_id in output:
            raise DownstreamDataError(
                f"Duplicate corpus skill ID: path={path}, skill_id={skill_id}"
            )
        output[skill_id] = skill
    return output


def load_decisions(path: Path) -> dict[str, JsonObject]:
    """Load unique retrieval decisions indexed by instance_id."""

    payload: JsonObject = require_object(
        load_json(path),
        f"decision-source:{path}",
    )
    values: list[JsonValue] = require_list(
        payload.get("results"),
        f"decision-source:{path}.results",
    )
    output: dict[str, JsonObject] = {}
    for index, value in enumerate(values):
        row: JsonObject = require_object(
            value,
            f"decision-source:{path}.results[{index}]",
        )
        instance_id: str = require_string(
            row.get("instance_id"),
            f"decision-source:{path}.results[{index}].instance_id",
        )
        if instance_id in output:
            raise DownstreamDataError(
                "Duplicate decision-source instance: "
                f"path={path}, instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def load_legacy_lines(path: Path) -> dict[str, LegacyLine]:
    """Load unique legacy answer records and preserve source lines."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Legacy JSONL file does not exist: path={path}"
        )
    output: dict[str, LegacyLine] = {}
    with path.open(encoding="utf-8", newline="") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue
            try:
                raw_record: JsonValue = cast(JsonValue, json.loads(raw_line))
            except json.JSONDecodeError as error:
                raise DownstreamDataError(
                    "Legacy JSONL is malformed: "
                    f"path={path}, line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            record: JsonObject = require_object(
                raw_record,
                f"legacy:{path}:{line_number}",
            )
            instance_id: str = require_string(
                record.get("instance_id"),
                f"legacy:{path}:{line_number}.instance_id",
            )
            if instance_id in output:
                raise DownstreamDataError(
                    "Legacy JSONL contains duplicate instance ID: "
                    f"path={path}, instance_id={instance_id}"
                )
            output[instance_id] = {
                "line_number": line_number,
                "raw_line": raw_line,
                "line_sha256": sha256_text(raw_line.rstrip("\r\n")),
                "record": record,
            }
    return output


def load_manifest(
    path: Path,
    instances_sha256: str,
    corpus_sha256: str,
) -> RuntimeManifest:
    """Load and validate one answer runtime manifest."""

    raw_manifest: JsonValue = load_json(path)
    manifest_object: JsonObject = require_object(
        raw_manifest,
        f"runtime-manifest:{path}",
    )
    raw_code_hash: JsonValue | None = manifest_object.get(
        "answer_code_bundle_sha256"
    )
    code_hash: str = require_string(
        raw_code_hash,
        f"runtime-manifest:{path}.answer_code_bundle_sha256",
    )
    return validate_answer_runtime_manifest(
        manifest_object,
        instances_sha256,
        corpus_sha256,
        code_hash,
    )


def expected_skill_ids(
    arm: SemanticArm,
    decision: Mapping[str, JsonLike],
) -> tuple[str, ...]:
    """Return the expected decision for one semantic arm."""

    raw_retrieved: JsonValue | None = decision.get("retrieved")
    retrieved_values: list[JsonValue] = require_list(
        raw_retrieved,
        f"decision:{decision.get('instance_id')}.retrieved",
    )
    retrieved: list[JsonObject] = [
        require_object(
            value,
            f"decision:{decision.get('instance_id')}.retrieved[{index}]",
        )
        for index, value in enumerate(retrieved_values)
    ]
    if arm == "routed_always":
        return always_expected_skill_ids(retrieved)
    return gated_expected_skill_ids(retrieved)


def loaded_skills(
    skill_ids: Sequence[str],
    corpus: Mapping[str, JsonObject],
    instance_id: str,
) -> list[JsonObject]:
    """Resolve an ordered loading decision against the frozen corpus."""

    output: list[JsonObject] = []
    for skill_id in skill_ids:
        if skill_id not in corpus:
            raise DownstreamDataError(
                "Answer decision references an unknown skill: "
                f"instance_id={instance_id}, skill_id={skill_id}"
            )
        output.append(corpus[skill_id])
    return output


def answer_generation(
    runtime: AnswerRuntime,
    served_model: str,
) -> AnswerGeneration:
    """Build the frozen answer generation identity."""

    extra_body: JsonLike = runtime["get_extra_body"](served_model, THINKING)
    normalized_extra_body: JsonValue = cast(
        JsonValue,
        json.loads(canonical_json(extra_body)),
    )
    return {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "extra_body": normalized_extra_body,
    }


def answer_hash(
    arm: SemanticArm,
    instance: JsonObject,
    skills: Sequence[JsonObject],
    manifest: RuntimeManifest,
    runtime: AnswerRuntime,
) -> str:
    """Render and hash one exact direct-answer request."""

    instance_id: str = require_string(
        instance.get("instance_id"),
        "answer.instance_id",
    )
    skill_texts: list[str] = []
    tools: list[Mapping[str, JsonLike]] = []
    for skill in skills:
        content: JsonValue | None = skill.get("content")
        if not isinstance(content, str):
            raise DownstreamDataError(
                "Skill content must be a string: "
                f"instance_id={instance_id}, skill_id={skill.get('skill_id')!r}"
            )
        if content:
            skill_texts.append(content)
        raw_tools: JsonValue | None = skill.get("tools", [])
        tool_values: list[JsonValue] = require_list(
            raw_tools,
            f"skill:{skill.get('skill_id')}.tools",
        )
        for tool_index, raw_tool in enumerate(tool_values):
            tools.append(
                require_object(
                    raw_tool,
                    f"skill:{skill.get('skill_id')}.tools[{tool_index}]",
                )
            )
    system, user = runtime["build_prompt"](instance, skill_texts)
    messages: list[Mapping[str, JsonLike]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    served_model: str = require_string(
        manifest["runtime_identity"].get("served_model"),
        "runtime_identity.served_model",
    )
    return answer_execution_fingerprint(
        ANSWER_SCHEMA_VERSION,
        arm,
        instance_id,
        instance,
        messages,
        skills,
        tools,
        manifest["instances_sha256"],
        manifest["corpus_sha256"],
        manifest["runtime_identity"],
        answer_generation(runtime, served_model),
        manifest["answer_code_bundle_sha256"],
    )


def write_jsonl_atomic(
    path: Path,
    records: Sequence[Mapping[str, JsonLike]],
) -> None:
    """Atomically replace one JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(canonical_json(record) + "\n")
    temporary_path.replace(path)


def write_preseed_atomic(
    path: Path,
    legacy_lines: Sequence[LegacyLine],
) -> None:
    """Copy accepted source records without changing their JSON fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        for legacy_line in legacy_lines:
            raw_line: str = legacy_line["raw_line"]
            output_file.write(raw_line)
            if not raw_line.endswith("\n"):
                output_file.write("\n")
    temporary_path.replace(path)


def write_json_atomic(path: Path, payload: JsonLike) -> None:
    """Atomically replace one formatted JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def rejected_eligibility(reason: str) -> PreseedEligibility:
    """Return a named rejected eligibility result."""

    return PreseedEligibility(False, reason)


def main() -> None:
    """Audit every instance and emit preseed plus pending inputs."""

    args = parse_args()
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    decision_source_path: Path = cast(Path, args.decision_source).resolve()
    legacy_path: Path = cast(Path, args.legacy_jsonl).resolve()
    old_manifest_path: Path = cast(Path, args.old_runtime_manifest).resolve()
    new_manifest_path: Path = cast(Path, args.new_runtime_manifest).resolve()
    audit_output_path: Path = cast(Path, args.audit_output).resolve()
    preseed_output_path: Path = cast(Path, args.preseed_output).resolve()
    pending_output_path: Path = cast(Path, args.pending_output).resolve()
    result_tag: str = str(args.result_tag)
    arm: SemanticArm = cast(SemanticArm, str(args.arm))
    domain: str = str(args.domain)

    runtime: AnswerRuntime = load_answer_runtime()
    instances: list[JsonObject] = load_instances(instances_path, domain)
    instance_index: dict[str, JsonObject] = {
        require_string(instance.get("instance_id"), "instance.instance_id"): instance
        for instance in instances
    }
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    decisions: dict[str, JsonObject] = load_decisions(decision_source_path)
    decision_coverage = audit_record_coverage(
        list(instance_index),
        list(decisions),
    )
    if not decision_coverage.complete:
        raise DownstreamDataError(
            "Decision-source coverage mismatch: "
            f"missing={list(decision_coverage.missing_ids)[:20]}, "
            f"unexpected={list(decision_coverage.unexpected_ids)[:20]}"
        )
    instances_sha256: str = sha256_file(instances_path)
    corpus_sha256: str = sha256_file(corpus_path)
    old_manifest: RuntimeManifest = load_manifest(
        old_manifest_path,
        instances_sha256,
        corpus_sha256,
    )
    new_manifest: RuntimeManifest = load_manifest(
        new_manifest_path,
        instances_sha256,
        corpus_sha256,
    )
    legacy_lines: dict[str, LegacyLine] = load_legacy_lines(legacy_path)
    source_jsonl_sha256: str = sha256_file(legacy_path)
    expected_legacy_label: str | None = allowed_legacy_label(result_tag, arm)
    if legacy_lines:
        if expected_legacy_label is None:
            raise DownstreamDataError(
                "This model/arm has no permitted legacy reuse source: "
                f"result_tag={result_tag}, arm={arm}, "
                f"legacy_records={len(legacy_lines)}"
            )
        require_runtime_code_files(new_manifest, "new-runtime-manifest")
        validate_legacy_manifest_evidence(
            old_manifest,
            source_jsonl_sha256,
            len(legacy_lines),
            result_tag,
            arm,
            expected_legacy_label,
        )
    unexpected_legacy_ids: list[str] = sorted(set(legacy_lines) - set(instance_index))
    if unexpected_legacy_ids:
        raise DownstreamDataError(
            "Legacy JSONL contains instances outside the current job: "
            f"sample={unexpected_legacy_ids[:20]}"
        )
    runtime_identity_matches: bool = (
        canonical_json(old_manifest["runtime_identity"])
        == canonical_json(new_manifest["runtime_identity"])
    )
    audit_records: list[AuditRecord] = []
    accepted_lines: list[LegacyLine] = []
    pending_instances: list[JsonObject] = []

    for instance in instances:
        instance_id: str = require_string(
            instance.get("instance_id"),
            "instance.instance_id",
        )
        new_skill_ids: tuple[str, ...] = expected_skill_ids(
            arm,
            decisions[instance_id],
        )
        new_skills: list[JsonObject] = loaded_skills(
            new_skill_ids,
            corpus,
            instance_id,
        )
        new_request_hash: str = answer_hash(
            arm,
            instance,
            new_skills,
            new_manifest,
            runtime,
        )
        legacy_line: LegacyLine | None = legacy_lines.get(instance_id)
        if legacy_line is None:
            audit_records.append(
                {
                    "instance_id": instance_id,
                    "arm": arm,
                    "legacy_label": expected_legacy_label,
                    "status": "needs_inference",
                    "reason": "legacy_record_missing",
                    "needs_inference": True,
                    "expected_skill_ids": list(new_skill_ids),
                    "legacy_skill_ids": [],
                    "new_request_hash": new_request_hash,
                    "old_request_hash": None,
                    "source_jsonl_sha256": source_jsonl_sha256,
                    "source_line_number": None,
                    "source_line_sha256": None,
                    "runtime_identity_matches": runtime_identity_matches,
                }
            )
            pending_instances.append(instance)
            continue
        legacy_record: JsonObject = legacy_line["record"]
        raw_method: JsonValue | None = legacy_record.get("method")
        legacy_skill_ids: tuple[str, ...] = ()
        old_request_hash: str | None = None
        if raw_method != expected_legacy_label:
            eligibility: PreseedEligibility = rejected_eligibility(
                "legacy_label_mismatch"
            )
        else:
            try:
                old_failure_category: FailureCategory = derive_legacy_answer_success(
                    legacy_record
                )
                legacy_skill_ids = normalize_skill_ids(
                    legacy_record.get("skill_ids_used"),
                    "skill_ids_used",
                )
                old_skills: list[JsonObject] = loaded_skills(
                    legacy_skill_ids,
                    corpus,
                    instance_id,
                )
                old_request_hash = answer_hash(
                    arm,
                    instance,
                    old_skills,
                    old_manifest,
                    runtime,
                )
                eligibility = same_arm_preseed_eligibility(
                    arm,
                    arm,
                    new_request_hash,
                    old_request_hash,
                    old_failure_category,
                    legacy_record.get("raw_output"),
                    legacy_skill_ids,
                    new_skill_ids,
                    runtime_identity_matches,
                )
            except DownstreamDataError as error:
                eligibility = rejected_eligibility(
                    f"legacy_record_invalid:{type(error).__name__}:{error}"
                )
        status: str = "reused_same_arm" if eligibility.eligible else "rejected"
        audit_records.append(
            {
                "instance_id": instance_id,
                "arm": arm,
                "legacy_label": expected_legacy_label,
                "status": status,
                "reason": eligibility.reason,
                "needs_inference": not eligibility.eligible,
                "expected_skill_ids": list(new_skill_ids),
                "legacy_skill_ids": list(legacy_skill_ids),
                "new_request_hash": new_request_hash,
                "old_request_hash": old_request_hash,
                "source_jsonl_sha256": source_jsonl_sha256,
                "source_line_number": legacy_line["line_number"],
                "source_line_sha256": legacy_line["line_sha256"],
                "runtime_identity_matches": runtime_identity_matches,
            }
        )
        if eligibility.eligible:
            accepted_lines.append(legacy_line)
        else:
            pending_instances.append(instance)

    write_jsonl_atomic(audit_output_path, audit_records)
    write_preseed_atomic(preseed_output_path, accepted_lines)
    write_json_atomic(pending_output_path, pending_instances)
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for record in audit_records:
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
        reason_counts[record["reason"]] = reason_counts.get(record["reason"], 0) + 1
    summary: JsonObject = {
        "event": "k2_reuse_audit_complete",
        "result_tag": result_tag,
        "domain": domain,
        "arm": arm,
        "records": len(audit_records),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "preseed_records": len(accepted_lines),
        "pending_records": len(pending_instances),
        "decision_source_sha256": sha256_file(decision_source_path),
        "legacy_source_sha256": source_jsonl_sha256,
        "old_runtime_manifest_sha256": sha256_file(old_manifest_path),
        "new_runtime_manifest_sha256": sha256_file(new_manifest_path),
        "legacy_evidence_required": bool(legacy_lines),
        "audit_output_sha256": sha256_file(audit_output_path),
        "preseed_output_sha256": sha256_file(preseed_output_path),
        "pending_output_sha256": sha256_file(pending_output_path),
    }
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
