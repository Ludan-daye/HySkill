#!/usr/bin/env python3
"""Validate K=2 selection or answer records against frozen request identity."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    FailureCategory,
    JsonLike,
    JsonObject,
    JsonValue,
    PreseedEligibility,
    RuntimeManifest,
    SemanticArm,
    allowed_legacy_label,
    audit_record_coverage,
    canonical_json,
    derive_legacy_answer_success,
    normalize_skill_ids,
    require_runtime_code_files,
    same_arm_preseed_eligibility,
    select_expected_skill_ids,
    sha256_file,
    sha256_text,
    validate_legacy_manifest_evidence,
    validate_failure_category,
    validate_selector_runtime_manifest,
)
from scripts.audit_k2_reuse import (
    AnswerRuntime,
    LegacyLine,
    answer_hash,
    expected_skill_ids,
    load_answer_runtime,
    load_corpus,
    load_decisions,
    load_instances,
    load_legacy_lines,
    load_manifest,
    loaded_skills,
    require_list,
    require_object,
    require_string,
)
from scripts.run_k2_answers import AnswerLine, indexed_answer_lines, load_jsonl
from scripts.run_select_only import (
    POOL_SIZE,
    SELECTION_RECORD_SCHEMA_VERSION,
    SelectorInput,
    SelectionRecord,
    SelectorRuntime,
    build_selector_inputs,
    load_json,
    load_selector_runtime,
    object_rows,
    read_selection_records,
    require_json_list,
    require_json_object,
)


class IndependentReuseExpectation(TypedDict):
    """Validator-owned recomputation of one reuse-audit row."""

    status: str
    reason: str
    needs_inference: bool
    legacy_skill_ids: list[str]
    old_request_hash: str | None
    source_line_number: int | None
    source_line_sha256: str | None


def add_selection_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the selection-stage validator."""

    parser = subparsers.add_parser("selection")
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--selected-source", required=True, type=Path)
    parser.add_argument("--attempt-log", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--code-bundle-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.set_defaults(func=validate_selection)


def add_answer_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the answer-stage validator."""

    parser = subparsers.add_parser("answer")
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--decision-source", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--legacy-jsonl", required=True, type=Path)
    parser.add_argument("--old-runtime-manifest", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--model", required=True)
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
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.set_defaults(func=validate_answers)


def parse_args() -> argparse.Namespace:
    """Parse an explicit validation stage."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    add_selection_parser(subparsers)
    add_answer_parser(subparsers)
    return parser.parse_args()


def write_json_atomic(path: Path, payload: Mapping[str, JsonLike]) -> None:
    """Atomically write one validation report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def selection_input_index(
    selector_inputs: Sequence[SelectorInput],
) -> dict[str, SelectorInput]:
    """Index unique selector inputs by instance ID."""

    output: dict[str, SelectorInput] = {}
    for selector_input in selector_inputs:
        instance_id: str = require_string(
            selector_input["instance"].get("instance_id"),
            "selector-input.instance_id",
        )
        if instance_id in output:
            raise DownstreamDataError(
                f"Duplicate selector input: instance_id={instance_id}"
            )
        output[instance_id] = selector_input
    return output


def load_attempt_records(path: Path) -> list[JsonObject]:
    """Load a structured attempt log."""

    if not path.is_file():
        raise FileNotFoundError(f"Attempt log does not exist: path={path}")
    output: list[JsonObject] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_record: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise DownstreamDataError(
                    "Attempt log is malformed: "
                    f"path={path}, line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            output.append(
                require_object(
                    raw_record,
                    f"attempt-log:{path}:{line_number}",
                )
            )
    return output


def validate_selection_record(
    record: SelectionRecord,
    selector_input: SelectorInput,
    manifest: RuntimeManifest,
    code_bundle_sha256_value: str,
    model: str,
    domain: str,
    source_sha256: str,
) -> FailureCategory:
    """Validate one selection record against its rendered request."""

    instance_id: str = require_string(
        record.get("instance_id"),
        "selection.instance_id",
    )
    expected_candidate_ids: list[str] = [
        candidate["skill_id"] for candidate in selector_input["candidate_displays"]
    ]
    if record.get("schema_version") != SELECTION_RECORD_SCHEMA_VERSION:
        raise DownstreamDataError(
            "Selection schema mismatch: "
            f"instance_id={instance_id}, actual={record.get('schema_version')!r}"
        )
    expected_fields: tuple[tuple[str, JsonLike, JsonLike], ...] = (
        ("dataset", domain, record.get("dataset")),
        ("arm", "routed_select", record.get("arm")),
        ("model", model, record.get("model")),
        (
            "ordered_candidate_ids",
            expected_candidate_ids,
            record.get("ordered_candidate_ids"),
        ),
        ("candidate_hash", selector_input["candidate_hash"], record.get("candidate_hash")),
        (
            "selector_request_hash",
            selector_input["request_hash"],
            record.get("selector_request_hash"),
        ),
        ("runtime_identity", manifest["runtime_identity"], record.get("runtime_identity")),
        (
            "code_bundle_sha256",
            code_bundle_sha256_value,
            record.get("code_bundle_sha256"),
        ),
        ("source_sha256", source_sha256, record.get("source_sha256")),
    )
    for field_name, expected_value, actual_value in expected_fields:
        if canonical_json(actual_value) != canonical_json(expected_value):
            raise DownstreamDataError(
                "Selection identity mismatch: "
                f"instance_id={instance_id}, field={field_name}, "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    category: FailureCategory = validate_failure_category(
        record.get("failure_category")
    )
    raw_responses_value: list[JsonValue] = require_list(
        record.get("raw_responses"),
        f"selection:{instance_id}.raw_responses",
    )
    raw_responses: list[str] = [
        response
        for response in raw_responses_value
        if isinstance(response, str)
    ]
    if len(raw_responses) != len(raw_responses_value):
        raise DownstreamDataError(
            f"Selection raw_responses contains non-string values: instance_id={instance_id}"
        )
    selected_skill_id: JsonValue | None = record.get("selected_skill_id")
    if selected_skill_id is not None and not isinstance(selected_skill_id, str):
        raise DownstreamDataError(
            "Selected skill ID must be a string or null: "
            f"instance_id={instance_id}, value={selected_skill_id!r}"
        )
    selected_rank: JsonValue | None = record.get("selected_rank")
    expected_loaded: tuple[str, ...] = select_expected_skill_ids(
        selected_skill_id,
        category,
    )
    if category in ("success", "selector_fallback"):
        if not isinstance(selected_rank, int) or isinstance(selected_rank, bool):
            raise DownstreamDataError(
                f"Selected rank must be an integer: instance_id={instance_id}"
            )
        if selected_rank < 1 or selected_rank > POOL_SIZE:
            raise DownstreamDataError(
                "Selected rank is outside the frozen pool: "
                f"instance_id={instance_id}, rank={selected_rank}"
            )
        if expected_candidate_ids[selected_rank - 1] != expected_loaded[0]:
            raise DownstreamDataError(
                "Selected rank and skill ID disagree: "
                f"instance_id={instance_id}, rank={selected_rank}, "
                f"skill_id={expected_loaded[0]}"
            )
    if category == "success":
        if record.get("parse_success") is not True:
            raise DownstreamDataError(
                f"Successful selector did not record parse_success: instance_id={instance_id}"
            )
        if record.get("rank1_fallback") is not False:
            raise DownstreamDataError(
                f"Successful selector incorrectly marks fallback: instance_id={instance_id}"
            )
        if not raw_responses:
            raise DownstreamDataError(
                f"Successful selector has no raw response: instance_id={instance_id}"
            )
    elif category == "selector_fallback":
        if selected_rank != 1 or record.get("rank1_fallback") is not True:
            raise DownstreamDataError(
                f"Selector fallback is not rank-1: instance_id={instance_id}"
            )
        if record.get("parse_success") is not False or len(raw_responses) != 3:
            raise DownstreamDataError(
                "Selector fallback does not preserve three failed parses: "
                f"instance_id={instance_id}, responses={len(raw_responses)}"
            )
    else:
        if selected_skill_id is not None or selected_rank is not None:
            raise DownstreamDataError(
                f"Failed selector contains a selection: instance_id={instance_id}"
            )
        if not isinstance(record.get("error"), dict):
            raise DownstreamDataError(
                f"Failed selector lacks structured error: instance_id={instance_id}"
            )
    return category


def validate_selected_source(
    path: Path,
    source_instance_ids: Sequence[str],
    selection_records: Mapping[str, SelectionRecord],
) -> None:
    """Validate the top-1 source emitted by selection-only."""

    payload: dict[str, JsonValue] = require_json_object(load_json(path), path)
    raw_results: JsonValue = payload.get("results")
    if not isinstance(raw_results, list):
        raise DownstreamDataError(
            f"Selected source does not contain a results list: path={path}"
        )
    rows: list[JsonObject] = object_rows(raw_results, path)
    row_index: dict[str, JsonObject] = {}
    for row in rows:
        instance_id: str = require_string(
            row.get("instance_id"),
            "selected-source.instance_id",
        )
        if instance_id in row_index:
            raise DownstreamDataError(
                f"Selected source has duplicate instance: instance_id={instance_id}"
            )
        row_index[instance_id] = row
    coverage = audit_record_coverage(source_instance_ids, list(row_index))
    if not coverage.complete:
        raise DownstreamDataError(
            "Selected-source coverage mismatch: "
            f"missing={list(coverage.missing_ids)[:20]}, "
            f"unexpected={list(coverage.unexpected_ids)[:20]}"
        )
    for instance_id in source_instance_ids:
        category: FailureCategory = validate_failure_category(
            selection_records[instance_id].get("failure_category")
        )
        expected: tuple[str, ...] = select_expected_skill_ids(
            cast(str | None, selection_records[instance_id].get("selected_skill_id")),
            category,
        )
        retrieved_values: list[JsonValue] = require_list(
            row_index[instance_id].get("retrieved"),
            f"selected-source:{instance_id}.retrieved",
        )
        actual: tuple[str, ...] = tuple(
            require_string(
                require_object(
                    value,
                    f"selected-source:{instance_id}.retrieved[{index}]",
                ).get("skill_id"),
                f"selected-source:{instance_id}.retrieved[{index}].skill_id",
            )
            for index, value in enumerate(retrieved_values)
        )
        if actual != expected:
            raise DownstreamDataError(
                "Selected-source decision mismatch: "
                f"instance_id={instance_id}, expected={expected}, actual={actual}"
            )
    metadata: JsonObject = require_object(
        payload.get("metadata"),
        "selected-source.metadata",
    )
    selector_metadata: JsonObject = require_object(
        metadata.get("selector"),
        "selected-source.metadata.selector",
    )
    if selector_metadata.get("complete") is not True:
        raise DownstreamDataError(
            "Selected source is not marked complete: "
            f"metadata={selector_metadata}"
        )


def validate_selection(args: argparse.Namespace) -> None:
    """Validate one selection-only model-domain job."""

    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    source_path: Path = cast(Path, args.source).resolve()
    selection_path: Path = cast(Path, args.selection).resolve()
    selected_source_path: Path = cast(Path, args.selected_source).resolve()
    attempt_log_path: Path = cast(Path, args.attempt_log).resolve()
    manifest_path: Path = cast(Path, args.runtime_manifest).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    code_bundle_sha256_value: str = str(args.code_bundle_sha256)
    model: str = str(args.model)
    domain: str = str(args.domain)
    expected_count: int = int(args.expected_count)

    runtime: SelectorRuntime = load_selector_runtime()
    instances: list[JsonObject] = object_rows(
        require_json_list(load_json(instances_path), instances_path),
        instances_path,
    )
    if len(instances) != expected_count:
        raise DownstreamDataError(
            "Selection instance denominator mismatch: "
            f"expected={expected_count}, actual={len(instances)}"
        )
    for instance in instances:
        if instance.get("dataset") != domain:
            raise DownstreamDataError(
                "Selection instance domain mismatch: "
                f"instance_id={instance.get('instance_id')!r}, "
                f"expected={domain}, actual={instance.get('dataset')!r}"
            )
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    source_payload: dict[str, JsonValue] = require_json_object(
        load_json(source_path),
        source_path,
    )
    raw_source_rows: JsonValue = source_payload.get("results")
    if not isinstance(raw_source_rows, list):
        raise DownstreamDataError(
            f"Selection source must contain results list: path={source_path}"
        )
    source_rows: list[JsonObject] = object_rows(raw_source_rows, source_path)
    source_index: dict[str, JsonObject] = {
        require_string(row.get("instance_id"), "source.instance_id"): row
        for row in source_rows
    }
    manifest_raw: JsonValue = load_json(manifest_path)
    manifest: RuntimeManifest = validate_selector_runtime_manifest(
        manifest_raw,
        sha256_file(instances_path),
        sha256_file(corpus_path),
        code_bundle_sha256_value,
    )
    if manifest["runtime_identity"].get("served_model") != model:
        raise DownstreamDataError(
            "Selection model identity mismatch: "
            f"cli_model={model}, "
            f"manifest={manifest['runtime_identity'].get('served_model')!r}"
        )
    selector_inputs: dict[str, SelectorInput] = selection_input_index(
        build_selector_inputs(
            instances,
            source_index,
            corpus,
            runtime,
            manifest,
            code_bundle_sha256_value,
        )
    )
    records: list[SelectionRecord] = read_selection_records(selection_path)
    record_index: dict[str, SelectionRecord] = {
        record["instance_id"]: record for record in records
    }
    coverage = audit_record_coverage(
        list(selector_inputs),
        [record["instance_id"] for record in records],
    )
    if not coverage.complete:
        raise DownstreamDataError(
            "Selection coverage mismatch: "
            f"missing={list(coverage.missing_ids)[:20]}, "
            f"duplicates={list(coverage.duplicate_ids)[:20]}, "
            f"unexpected={list(coverage.unexpected_ids)[:20]}"
        )
    source_sha256: str = sha256_file(source_path)
    category_counts: dict[str, int] = {}
    for record in records:
        instance_id: str = record["instance_id"]
        category: FailureCategory = validate_selection_record(
            record,
            selector_inputs[instance_id],
            manifest,
            code_bundle_sha256_value,
            model,
            domain,
            source_sha256,
        )
        category_counts[category] = category_counts.get(category, 0) + 1
    validate_selected_source(
        selected_source_path,
        list(selector_inputs),
        record_index,
    )
    attempts: list[JsonObject] = load_attempt_records(attempt_log_path)
    known_request_hashes: set[str] = {
        selector_input["request_hash"] for selector_input in selector_inputs.values()
    }
    unknown_attempt_hashes: list[str] = sorted(
        {
            require_string(
                attempt.get("selector_request_hash"),
                "attempt.selector_request_hash",
            )
            for attempt in attempts
        }
        - known_request_hashes
    )
    if unknown_attempt_hashes:
        raise DownstreamDataError(
            "Attempt log contains unknown request hashes: "
            f"sample={unknown_attempt_hashes[:20]}"
        )
    unresolved: int = (
        category_counts.get("infra_transient", 0)
        + category_counts.get("unclassified_error", 0)
    )
    report: JsonObject = {
        "schema_version": "k2-selection-validation-v1",
        "valid": unresolved == 0,
        "model": model,
        "domain": domain,
        "expected": expected_count,
        "observed": len(records),
        "failure_categories": category_counts,
        "attempt_records": len(attempts),
        "selection_sha256": sha256_file(selection_path),
        "selected_source_sha256": sha256_file(selected_source_path),
        "attempt_log_sha256": sha256_file(attempt_log_path),
    }
    write_json_atomic(output_path, report)
    print(canonical_json(report))
    if unresolved:
        raise SystemExit(1)


def load_raw_audit(path: Path) -> dict[str, JsonObject]:
    """Load the complete reuse sidecar indexed by instance ID."""

    lines: list[AnswerLine] = load_jsonl(path, "audit")
    output: dict[str, JsonObject] = {}
    for line in lines:
        record: JsonObject = line["record"]
        instance_id: str = require_string(
            record.get("instance_id"),
            "audit.instance_id",
        )
        if instance_id in output:
            raise DownstreamDataError(
                f"Reuse audit contains duplicate instance: instance_id={instance_id}"
            )
        output[instance_id] = record
    return output


def validate_reused_answer(
    instance_id: str,
    answer_line: AnswerLine,
    legacy_line: LegacyLine,
    expected: tuple[str, ...],
) -> None:
    """Validate one unchanged preseed directly against the legacy source."""

    expected_line_sha: str = legacy_line["line_sha256"]
    actual_line_sha: str = sha256_text(
        answer_line["raw_line"].rstrip("\r\n")
    )
    if actual_line_sha != expected_line_sha:
        raise DownstreamDataError(
            "Reused answer line hash mismatch: "
            f"instance_id={instance_id}, expected={expected_line_sha}, "
            f"actual={actual_line_sha}"
        )
    record: JsonObject = answer_line["record"]
    derive_legacy_answer_success(record)
    actual_skill_ids: tuple[str, ...] = normalize_skill_ids(
        record.get("skill_ids_used"),
        "skill_ids_used",
    )
    if actual_skill_ids != expected:
        raise DownstreamDataError(
            "Reused answer loading mismatch: "
            f"instance_id={instance_id}, expected={expected}, actual={actual_skill_ids}"
        )


def validate_new_answer(
    instance_id: str,
    answer_line: AnswerLine,
    audit_row: Mapping[str, JsonLike],
    expected: tuple[str, ...],
    manifest: RuntimeManifest,
    model: str,
    arm: SemanticArm,
    domain: str,
    decision_source_sha256: str,
) -> FailureCategory:
    """Validate one newly generated answer record."""

    record: JsonObject = answer_line["record"]
    expected_fields: tuple[tuple[str, JsonLike, JsonLike], ...] = (
        ("schema_version", "k2-answer-record-v1", record.get("schema_version")),
        ("dataset", domain, record.get("dataset")),
        ("method", arm, record.get("method")),
        ("served_model", model, record.get("served_model")),
        (
            "expected_skill_ids",
            list(expected),
            record.get("expected_skill_ids"),
        ),
        (
            "request_hash",
            audit_row.get("new_request_hash"),
            record.get("request_hash"),
        ),
        (
            "runtime_identity",
            manifest["runtime_identity"],
            record.get("runtime_identity"),
        ),
        (
            "answer_code_bundle_sha256",
            manifest["answer_code_bundle_sha256"],
            record.get("answer_code_bundle_sha256"),
        ),
        (
            "decision_source_sha256",
            decision_source_sha256,
            record.get("decision_source_sha256"),
        ),
    )
    for field_name, expected_value, actual_value in expected_fields:
        if canonical_json(actual_value) != canonical_json(expected_value):
            raise DownstreamDataError(
                "New answer identity mismatch: "
                f"instance_id={instance_id}, field={field_name}, "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    category: FailureCategory = validate_failure_category(
        record.get("failure_category")
    )
    if category == "selector_fallback":
        raise DownstreamDataError(
            f"Answer record cannot use selector_fallback category: instance_id={instance_id}"
        )
    raw_output: JsonValue | None = record.get("raw_output")
    actual_skill_ids: tuple[str, ...] = normalize_skill_ids(
        record.get("skill_ids_used"),
        "skill_ids_used",
    )
    actual_injection: JsonObject = require_object(
        record.get("actual_injection_state"),
        f"answer:{instance_id}.actual_injection_state",
    )
    if category == "success":
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise DownstreamDataError(
                f"Successful answer has empty raw_output: instance_id={instance_id}"
            )
        if actual_skill_ids != expected:
            raise DownstreamDataError(
                "Successful answer injection mismatch: "
                f"instance_id={instance_id}, expected={expected}, actual={actual_skill_ids}"
            )
        if actual_injection.get("state") != "confirmed_by_engine":
            raise DownstreamDataError(
                f"Successful answer lacks confirmed injection state: instance_id={instance_id}"
            )
    else:
        if not isinstance(record.get("error"), dict):
            raise DownstreamDataError(
                f"Failed answer lacks structured error: instance_id={instance_id}"
            )
        if actual_injection.get("state") != "request_submitted":
            raise DownstreamDataError(
                f"Failed answer lacks submitted injection state: instance_id={instance_id}"
            )
        injected_values: list[JsonValue] = require_list(
            actual_injection.get("skill_ids"),
            f"answer:{instance_id}.actual_injection_state.skill_ids",
        )
        injected: tuple[str, ...] = tuple(
            require_string(
                skill_id,
                f"answer:{instance_id}.actual_injection_state.skill_ids[{index}]",
            )
            for index, skill_id in enumerate(injected_values)
        )
        if injected != expected:
            raise DownstreamDataError(
                "Failed answer expected injection mismatch: "
                f"instance_id={instance_id}, expected={expected}, actual={injected}"
            )
    return category


def independent_reuse_expectation(
    arm: SemanticArm,
    instance: JsonObject,
    expected_skill_ids_value: tuple[str, ...],
    legacy_line: LegacyLine | None,
    expected_legacy_label: str | None,
    corpus: Mapping[str, JsonObject],
    old_manifest: RuntimeManifest,
    new_request_hash: str,
    runtime: AnswerRuntime,
    runtime_identity_matches: bool,
) -> IndependentReuseExpectation:
    """Recompute one reuse decision without trusting the audit sidecar."""

    if legacy_line is None:
        return {
            "status": "needs_inference",
            "reason": "legacy_record_missing",
            "needs_inference": True,
            "legacy_skill_ids": [],
            "old_request_hash": None,
            "source_line_number": None,
            "source_line_sha256": None,
        }
    legacy_record: JsonObject = legacy_line["record"]
    legacy_skill_ids: tuple[str, ...] = ()
    old_request_hash: str | None = None
    if legacy_record.get("method") != expected_legacy_label:
        eligibility: PreseedEligibility = PreseedEligibility(
            False,
            "legacy_label_mismatch",
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
                require_string(
                    instance.get("instance_id"),
                    "instance.instance_id",
                ),
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
                expected_skill_ids_value,
                runtime_identity_matches,
            )
        except DownstreamDataError as error:
            eligibility = PreseedEligibility(
                False,
                f"legacy_record_invalid:{type(error).__name__}:{error}",
            )
    return {
        "status": "reused_same_arm" if eligibility.eligible else "rejected",
        "reason": eligibility.reason,
        "needs_inference": not eligibility.eligible,
        "legacy_skill_ids": list(legacy_skill_ids),
        "old_request_hash": old_request_hash,
        "source_line_number": legacy_line["line_number"],
        "source_line_sha256": legacy_line["line_sha256"],
    }


def validate_audit_row_independently(
    instance_id: str,
    audit_row: Mapping[str, JsonLike],
    arm: SemanticArm,
    expected_legacy_label: str | None,
    expected_skill_ids_value: tuple[str, ...],
    new_request_hash: str,
    source_jsonl_sha256: str,
    runtime_identity_matches: bool,
    expectation: IndependentReuseExpectation,
) -> None:
    """Compare every provenance-bearing audit field to recomputed evidence."""

    expected_fields: tuple[tuple[str, JsonLike], ...] = (
        ("arm", arm),
        ("legacy_label", expected_legacy_label),
        ("status", expectation["status"]),
        ("reason", expectation["reason"]),
        ("needs_inference", expectation["needs_inference"]),
        ("expected_skill_ids", list(expected_skill_ids_value)),
        ("legacy_skill_ids", expectation["legacy_skill_ids"]),
        ("new_request_hash", new_request_hash),
        ("old_request_hash", expectation["old_request_hash"]),
        ("source_jsonl_sha256", source_jsonl_sha256),
        ("source_line_number", expectation["source_line_number"]),
        ("source_line_sha256", expectation["source_line_sha256"]),
        ("runtime_identity_matches", runtime_identity_matches),
    )
    for field_name, expected_value in expected_fields:
        actual_value: JsonLike | None = audit_row.get(field_name)
        if canonical_json(actual_value) != canonical_json(expected_value):
            raise DownstreamDataError(
                "Reuse audit does not match independent recomputation: "
                f"instance_id={instance_id}, field={field_name}, "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )


def validate_answers(args: argparse.Namespace) -> None:
    """Validate one complete preseed-plus-new answer job."""

    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    decision_source_path: Path = cast(Path, args.decision_source).resolve()
    answers_path: Path = cast(Path, args.answers).resolve()
    audit_path: Path = cast(Path, args.audit).resolve()
    legacy_path: Path = cast(Path, args.legacy_jsonl).resolve()
    old_manifest_path: Path = cast(Path, args.old_runtime_manifest).resolve()
    manifest_path: Path = cast(Path, args.runtime_manifest).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    result_tag: str = str(args.result_tag)
    model: str = str(args.model)
    arm: SemanticArm = cast(SemanticArm, str(args.arm))
    domain: str = str(args.domain)
    expected_count: int = int(args.expected_count)

    runtime: AnswerRuntime = load_answer_runtime()
    instances: list[JsonObject] = load_instances(instances_path, domain)
    if len(instances) != expected_count:
        raise DownstreamDataError(
            "Answer instance denominator mismatch: "
            f"expected={expected_count}, actual={len(instances)}"
        )
    instance_index: dict[str, JsonObject] = {
        require_string(instance.get("instance_id"), "instance.instance_id"): instance
        for instance in instances
    }
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    decisions: dict[str, JsonObject] = load_decisions(decision_source_path)
    audit: dict[str, JsonObject] = load_raw_audit(audit_path)
    answer_lines: list[AnswerLine] = load_jsonl(answers_path, "answers")
    answer_index: dict[str, AnswerLine] = indexed_answer_lines(
        answer_lines,
        "answers",
    )
    for name, observed_ids in (
        ("decision-source", list(decisions)),
        ("reuse-audit", list(audit)),
        ("answers", list(answer_index)),
    ):
        coverage = audit_record_coverage(list(instance_index), observed_ids)
        if not coverage.complete:
            raise DownstreamDataError(
                f"{name} coverage mismatch: "
                f"missing={list(coverage.missing_ids)[:20]}, "
                f"duplicates={list(coverage.duplicate_ids)[:20]}, "
                f"unexpected={list(coverage.unexpected_ids)[:20]}"
            )
    instances_sha256: str = sha256_file(instances_path)
    corpus_sha256: str = sha256_file(corpus_path)
    manifest: RuntimeManifest = load_manifest(
        manifest_path,
        instances_sha256,
        corpus_sha256,
    )
    old_manifest: RuntimeManifest = load_manifest(
        old_manifest_path,
        instances_sha256,
        corpus_sha256,
    )
    if manifest["runtime_identity"].get("served_model") != model:
        raise DownstreamDataError(
            "Answer model identity mismatch: "
            f"cli_model={model}, "
            f"manifest={manifest['runtime_identity'].get('served_model')!r}"
        )
    decision_source_sha256: str = sha256_file(decision_source_path)
    legacy_source_sha256: str = sha256_file(legacy_path)
    legacy_lines: dict[str, LegacyLine] = load_legacy_lines(legacy_path)
    expected_legacy_label: str | None = allowed_legacy_label(result_tag, arm)
    if legacy_lines:
        if expected_legacy_label is None:
            raise DownstreamDataError(
                "This model/arm has no permitted legacy reuse source: "
                f"result_tag={result_tag}, arm={arm}, "
                f"legacy_records={len(legacy_lines)}"
            )
        require_runtime_code_files(manifest, "new-runtime-manifest")
        validate_legacy_manifest_evidence(
            old_manifest,
            legacy_source_sha256,
            len(legacy_lines),
            result_tag,
            arm,
            expected_legacy_label,
        )
    unexpected_legacy_ids: list[str] = sorted(
        set(legacy_lines) - set(instance_index)
    )
    if unexpected_legacy_ids:
        raise DownstreamDataError(
            "Legacy JSONL contains instances outside the current job: "
            f"sample={unexpected_legacy_ids[:20]}"
        )
    runtime_identity_matches: bool = (
        canonical_json(old_manifest["runtime_identity"])
        == canonical_json(manifest["runtime_identity"])
    )
    category_counts: dict[str, int] = {}
    reused_count: int = 0
    for instance_id, instance in instance_index.items():
        audit_row: JsonObject = audit[instance_id]
        expected: tuple[str, ...] = expected_skill_ids(
            arm,
            decisions[instance_id],
        )
        skills: list[JsonObject] = loaded_skills(
            expected,
            corpus,
            instance_id,
        )
        request_hash: str = answer_hash(
            arm,
            instance,
            skills,
            manifest,
            runtime,
        )
        legacy_line: LegacyLine | None = legacy_lines.get(instance_id)
        expectation: IndependentReuseExpectation = independent_reuse_expectation(
            arm,
            instance,
            expected,
            legacy_line,
            expected_legacy_label,
            corpus,
            old_manifest,
            request_hash,
            runtime,
            runtime_identity_matches,
        )
        validate_audit_row_independently(
            instance_id,
            audit_row,
            arm,
            expected_legacy_label,
            expected,
            request_hash,
            legacy_source_sha256,
            runtime_identity_matches,
            expectation,
        )
        status: str = expectation["status"]
        if status == "reused_same_arm":
            if legacy_line is None:
                raise DownstreamDataError(
                    "Independent reuse expectation lacks a legacy line: "
                    f"instance_id={instance_id}"
                )
            reused_count += 1
            validate_reused_answer(
                instance_id,
                answer_index[instance_id],
                legacy_line,
                expected,
            )
            category: FailureCategory = "success"
        elif status in ("needs_inference", "rejected"):
            category = validate_new_answer(
                instance_id,
                answer_index[instance_id],
                audit_row,
                expected,
                manifest,
                model,
                arm,
                domain,
                decision_source_sha256,
            )
        else:
            raise DownstreamDataError(
                "Independent reuse computation produced an invalid status: "
                f"instance_id={instance_id}, status={status!r}"
            )
        category_counts[category] = category_counts.get(category, 0) + 1
    unresolved: int = (
        category_counts.get("infra_transient", 0)
        + category_counts.get("unclassified_error", 0)
    )
    report: JsonObject = {
        "schema_version": "k2-answer-validation-v2",
        "valid": unresolved == 0,
        "result_tag": result_tag,
        "model": model,
        "domain": domain,
        "arm": arm,
        "expected": expected_count,
        "observed": len(answer_lines),
        "reused_same_arm": reused_count,
        "new_records": len(answer_lines) - reused_count,
        "failure_categories": category_counts,
        "answers_sha256": sha256_file(answers_path),
        "audit_sha256": sha256_file(audit_path),
        "decision_source_sha256": decision_source_sha256,
        "legacy_source_sha256": legacy_source_sha256,
        "old_runtime_manifest_sha256": sha256_file(old_manifest_path),
        "runtime_manifest_sha256": sha256_file(manifest_path),
        "legacy_evidence_required": bool(legacy_lines),
    }
    write_json_atomic(output_path, report)
    print(canonical_json(report))
    if unresolved:
        raise SystemExit(1)


def main() -> None:
    """Dispatch to one explicit stage validator."""

    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
