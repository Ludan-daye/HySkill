#!/usr/bin/env python3
"""Run fresh BM25 plus native Select decisions with job-bound evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypeAlias, cast

from hyskill.runtime_matched_execution import (
    ExecutionContext,
    JobBoundManifest,
    JsonLike,
    OpenAIClientLike,
    bind_execution_context,
    canonical_json,
    classify_request_error,
    error_context,
    load_job_bound_manifest,
    manifest_artifact,
    sha256_file,
    validate_frozen_k2_runtime_reference,
    verify_job_bound_manifest_files,
    wrap_openai_client,
)
from hyskill.runtime_matched_select import (
    MAX_TOKENS,
    SELECT_ARM,
    SELECT_STAGE,
    SELECTION_RECORD_SCHEMA_VERSION,
    TEMPERATURE,
    THINKING,
    FailureCategory,
    JsonObject,
    JsonValue,
    NativeSelectorRuntime,
    PreparedSelection,
    SelectProtocolError,
    SelectionOutcome,
    SelectionRecord,
    SelectorRequestFailure,
    build_selection_record,
    load_native_selector_runtime,
    prepare_selection,
    require_list,
    require_object,
    require_select_eligible,
    require_string,
    run_selection_protocol,
)


RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0)
MAX_INFRA_ATTEMPTS: int = 3
BM25_SCHEMA_VERSION: str = "runtime-matched-bm25-v1"
AttemptPayload: TypeAlias = Mapping[str, JsonLike]


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-domain Select decision job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--bm25-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt-log", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--max-new-records", required=True, type=int)
    return parser.parse_args()


def load_json(path: Path, context: str) -> JsonValue:
    """Load one UTF-8 JSON file."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    try:
        return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise SelectProtocolError(
            f"{context} JSON is malformed: path={path}, "
            f"line={error.lineno}, column={error.colno}, message={error.msg}"
        ) from error


def index_rows(
    rows: Sequence[JsonObject],
    context: str,
) -> dict[str, JsonObject]:
    """Index rows by unique instance ID."""

    output: dict[str, JsonObject] = {}
    for row_number, row in enumerate(rows, start=1):
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}:{row_number}.instance_id",
        )
        if instance_id in output:
            raise SelectProtocolError(
                f"{context} contains duplicate instance ID: "
                f"instance_id={instance_id}"
            )
        output[instance_id] = row
    return output


def object_rows(
    values: Sequence[JsonValue],
    context: str,
) -> list[JsonObject]:
    """Validate a sequence of JSON object rows."""

    return [
        require_object(value, f"{context}[{index}]")
        for index, value in enumerate(values)
    ]


def load_instances(
    path: Path,
    domain: str,
    expected_count: int,
) -> list[JsonObject]:
    """Load one complete domain instance set."""

    rows: list[JsonObject] = object_rows(
        require_list(load_json(path, "instances"), "instances"),
        "instances",
    )
    if len(rows) != expected_count:
        raise SelectProtocolError(
            "Instance denominator mismatch: "
            f"expected={expected_count}, actual={len(rows)}, path={path}"
        )
    index_rows(rows, "instances")
    for row in rows:
        instance_id: str = require_string(
            row.get("instance_id"),
            "instance.instance_id",
        )
        instance_domain: str = require_string(
            row.get("dataset"),
            f"instance:{instance_id}.dataset",
        )
        if instance_domain != domain:
            raise SelectProtocolError(
                "Instance domain mismatch: "
                f"instance_id={instance_id}, expected={domain}, "
                f"actual={instance_domain}"
            )
    return rows


def load_corpus(path: Path) -> dict[str, JsonObject]:
    """Load a duplicate-free frozen skill corpus."""

    rows: list[JsonObject] = object_rows(
        require_list(load_json(path, "corpus"), "corpus"),
        "corpus",
    )
    output: dict[str, JsonObject] = {}
    for row_number, row in enumerate(rows, start=1):
        skill_id: str = require_string(
            row.get("skill_id"),
            f"corpus:{row_number}.skill_id",
        )
        if skill_id in output:
            raise SelectProtocolError(
                f"Corpus contains duplicate skill ID: skill_id={skill_id}"
            )
        output[skill_id] = row
    if not output:
        raise SelectProtocolError(f"Corpus is empty: path={path}")
    return output


def load_bm25_source(
    path: Path,
    expected_ids: set[str],
) -> dict[str, JsonObject]:
    """Load one exact-coverage frozen BM25 top-50 source."""

    payload: JsonObject = require_object(
        load_json(path, "BM25 source"),
        "BM25 source",
    )
    if payload.get("schema_version") != BM25_SCHEMA_VERSION:
        raise SelectProtocolError(
            "BM25 source schema mismatch: "
            f"expected={BM25_SCHEMA_VERSION}, "
            f"actual={payload.get('schema_version')!r}, path={path}"
        )
    rows: list[JsonObject] = object_rows(
        require_list(payload.get("results"), "BM25 source.results"),
        "BM25 source.results",
    )
    index: dict[str, JsonObject] = index_rows(rows, "BM25 source")
    observed_ids: set[str] = set(index)
    if observed_ids != expected_ids:
        raise SelectProtocolError(
            "BM25 source coverage mismatch: "
            f"missing={sorted(expected_ids - observed_ids)[:20]}, "
            f"unexpected={sorted(observed_ids - expected_ids)[:20]}"
        )
    return index


def manifest_object(
    value: JsonValue | None,
    context: str,
) -> JsonObject:
    """Return a manifest object using the local strict JSON type."""

    return require_object(value, context)


def validate_manifest_binding(
    manifest: JobBoundManifest,
    instances_path: Path,
    corpus_path: Path,
    bm25_path: Path,
    result_tag: str,
    served_model: str,
    domain: str,
    api_base: str,
    runtime: NativeSelectorRuntime,
) -> str:
    """Validate the manifest against this exact Select job."""

    facts: JsonObject = manifest_object(
        cast(JsonValue, manifest["runtime_facts"]),
        "manifest.runtime_facts",
    )
    job: JsonObject = manifest_object(
        facts.get("job"),
        "manifest.runtime_facts.job",
    )
    expected_job_fields: tuple[tuple[str, str], ...] = (
        ("result_tag", result_tag),
        ("model", served_model),
        ("domain", domain),
        ("arm", SELECT_ARM),
        ("stage", SELECT_STAGE),
    )
    for field_name, expected_value in expected_job_fields:
        actual_value: str = require_string(
            job.get(field_name),
            f"manifest.job.{field_name}",
        )
        if actual_value != expected_value:
            raise SelectProtocolError(
                "Runtime manifest job mismatch: "
                f"field={field_name}, expected={expected_value}, "
                f"actual={actual_value}"
            )
    endpoint: JsonObject = manifest_object(
        facts.get("endpoint"),
        "manifest.runtime_facts.endpoint",
    )
    endpoint_api_base: str = require_string(
        endpoint.get("api_base"),
        "manifest.endpoint.api_base",
    )
    if endpoint_api_base.rstrip("/") != api_base.rstrip("/"):
        raise SelectProtocolError(
            "CLI API base differs from the job-bound endpoint: "
            f"cli={api_base}, manifest={endpoint_api_base}"
        )
    endpoint_model: str = require_string(
        endpoint.get("served_model"),
        "manifest.endpoint.served_model",
    )
    if endpoint_model != served_model:
        raise SelectProtocolError(
            "CLI served model differs from the endpoint manifest: "
            f"cli={served_model}, manifest={endpoint_model}"
        )
    for artifact_name, path in (
        ("instances", instances_path),
        ("corpus", corpus_path),
        ("bm25_candidates", bm25_path),
    ):
        evidence = manifest_artifact(manifest, artifact_name)
        if Path(evidence["path"]).resolve() != path.resolve():
            raise SelectProtocolError(
                "Manifest artifact path mismatch: "
                f"name={artifact_name}, expected={path.resolve()}, "
                f"actual={evidence['path']}"
            )
        actual_size: int = path.stat().st_size
        actual_sha256: str = sha256_file(path)
        if (
            evidence["size_bytes"] != actual_size
            or evidence["sha256"] != actual_sha256
        ):
            raise SelectProtocolError(
                "Manifest artifact identity mismatch: "
                f"name={artifact_name}, expected_size={evidence['size_bytes']}, "
                f"actual_size={actual_size}, expected_sha={evidence['sha256']}, "
                f"actual_sha={actual_sha256}"
            )
    expected_generation: JsonObject = {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "extra_body": runtime["get_extra_body"](served_model, THINKING),
    }
    manifest_generation: JsonObject = manifest_object(
        cast(JsonValue, manifest["generation"]),
        "manifest.generation",
    )
    for field_name, expected_value in expected_generation.items():
        if manifest_generation.get(field_name) != expected_value:
            raise SelectProtocolError(
                "Selector generation differs from the job manifest: "
                f"field={field_name}, expected={expected_value!r}, "
                f"actual={manifest_generation.get(field_name)!r}"
            )
    return require_string(job.get("job_id"), "manifest.job.job_id")


def append_jsonl(
    path: Path,
    payload: AttemptPayload,
    write_lock: threading.Lock,
) -> None:
    """Append and flush one canonical JSONL event."""

    line: str = canonical_json(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with write_lock:
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(line)
            output_file.flush()


def write_records_atomic(
    path: Path,
    records: Sequence[SelectionRecord],
) -> None:
    """Atomically replace one decision JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(canonical_json(record))
            output_file.write("\n")
    temporary_path.replace(path)


def read_records(path: Path) -> list[SelectionRecord]:
    """Read existing fresh decisions for strict resume."""

    if not path.exists():
        return []
    records: list[SelectionRecord] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_record: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise SelectProtocolError(
                    "Decision JSONL is malformed: "
                    f"path={path}, line={line_number}, "
                    f"column={error.colno}, message={error.msg}"
                ) from error
            record: JsonObject = require_object(
                raw_record,
                f"decisions:{path}:{line_number}",
            )
            if record.get("schema_version") != SELECTION_RECORD_SCHEMA_VERSION:
                raise SelectProtocolError(
                    "Existing decision has unexpected schema: "
                    f"path={path}, line={line_number}, "
                    f"schema={record.get('schema_version')!r}"
                )
            instance_id: str = require_string(
                record.get("instance_id"),
                f"decisions:{path}:{line_number}.instance_id",
            )
            if instance_id in seen_ids:
                raise SelectProtocolError(
                    "Decision output contains duplicate instance: "
                    f"instance_id={instance_id}"
                )
            seen_ids.add(instance_id)
            records.append(cast(SelectionRecord, record))
    return records


def validate_existing_record(
    record: SelectionRecord,
    prepared: PreparedSelection,
    result_tag: str,
    served_model: str,
    domain: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    candidate_source_sha256: str,
) -> None:
    """Reject stale or cross-job resume records."""

    expected_fields: tuple[tuple[str, object], ...] = (
        ("model", result_tag),
        ("served_model", served_model),
        ("domain", domain),
        ("arm", SELECT_ARM),
        ("stage", SELECT_STAGE),
        ("candidate_hash", prepared["candidate_hash"]),
        ("selector_payload_hash", prepared["selector_payload_hash"]),
        (
            "execution_request_hash",
            prepared["execution_request_hash"],
        ),
        ("runtime_manifest_sha256", runtime_manifest_sha256),
        ("code_bundle_sha256", code_bundle_sha256),
        ("candidate_source_sha256", candidate_source_sha256),
        ("reused_same_arm", False),
    )
    for field_name, expected_value in expected_fields:
        actual_value: object = record.get(field_name)
        if actual_value != expected_value:
            raise SelectProtocolError(
                "Existing decision is stale or belongs to another job: "
                f"instance_id={record.get('instance_id')}, "
                f"field={field_name}, expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )


def main() -> None:
    """Run one resumable fresh Select decision job."""

    args = parse_args()
    workers: int = int(args.workers)
    expected_count: int = int(args.expected_count)
    max_new_records: int = int(args.max_new_records)
    if workers <= 0:
        raise ValueError(f"workers must be positive: workers={workers}")
    if expected_count <= 0:
        raise ValueError(
            f"expected-count must be positive: value={expected_count}"
        )
    if max_new_records < 0:
        raise ValueError(
            "max-new-records must be zero or positive: "
            f"value={max_new_records}"
        )
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    bm25_path: Path = cast(Path, args.bm25_source).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    attempt_log_path: Path = cast(Path, args.attempt_log).resolve()
    manifest_path: Path = cast(Path, args.runtime_manifest).resolve()
    repository_root: Path = cast(Path, args.repository_root).resolve()
    result_tag: str = str(args.result_tag)
    served_model: str = str(args.model)
    api_base: str = str(args.api_base)
    domain: str = str(args.domain)

    require_select_eligible(result_tag)
    runtime: NativeSelectorRuntime = load_native_selector_runtime()
    instances: list[JsonObject] = load_instances(
        instances_path,
        domain,
        expected_count,
    )
    instance_index: dict[str, JsonObject] = index_rows(
        instances,
        "instances",
    )
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    source_index: dict[str, JsonObject] = load_bm25_source(
        bm25_path,
        set(instance_index),
    )
    manifest: JobBoundManifest = load_job_bound_manifest(manifest_path)
    validate_frozen_k2_runtime_reference(manifest["runtime_facts"])
    verify_job_bound_manifest_files(manifest, repository_root)
    job_id: str = validate_manifest_binding(
        manifest,
        instances_path,
        corpus_path,
        bm25_path,
        result_tag,
        served_model,
        domain,
        api_base,
        runtime,
    )
    runtime_manifest_sha256: str = sha256_file(manifest_path)
    code_bundle_sha256: str = manifest["code_bundle_sha256"]
    candidate_source_sha256: str = sha256_file(bm25_path)
    prepared_inputs: list[PreparedSelection] = [
        prepare_selection(
            instance,
            source_index[
                require_string(
                    instance.get("instance_id"),
                    "instance.instance_id",
                )
            ],
            corpus,
            runtime,
            served_model,
            runtime_manifest_sha256,
            code_bundle_sha256,
        )
        for instance in instances
    ]
    prepared_index: dict[str, PreparedSelection] = {
        require_string(
            prepared["instance"].get("instance_id"),
            "prepared.instance_id",
        ): prepared
        for prepared in prepared_inputs
    }
    existing_records: list[SelectionRecord] = read_records(output_path)
    retained_records: list[SelectionRecord] = [
        record
        for record in existing_records
        if record.get("failure_category") != "infra_transient"
    ]
    if len(retained_records) != len(existing_records):
        write_records_atomic(output_path, retained_records)
    existing_index: dict[str, SelectionRecord] = {}
    for record in retained_records:
        instance_id: str = record["instance_id"]
        if instance_id not in prepared_index:
            raise SelectProtocolError(
                "Existing decision has unexpected instance: "
                f"instance_id={instance_id}"
            )
        validate_existing_record(
            record,
            prepared_index[instance_id],
            result_tag,
            served_model,
            domain,
            runtime_manifest_sha256,
            code_bundle_sha256,
            candidate_source_sha256,
        )
        existing_index[instance_id] = record

    write_lock: threading.Lock = threading.Lock()

    def usage_sink(event: Mapping[str, JsonLike]) -> None:
        append_jsonl(attempt_log_path, event, write_lock)

    raw_client: object = runtime["create_client"](api_base, None)
    client: object = wrap_openai_client(
        cast(OpenAIClientLike, raw_client),
        usage_sink,
    )
    extra_body: JsonObject | None = runtime["get_extra_body"](
        served_model,
        THINKING,
    )

    def run_one(prepared: PreparedSelection) -> SelectionRecord:
        instance_id: str = require_string(
            prepared["instance"].get("instance_id"),
            "prepared.instance_id",
        )
        client_call_attempts: int = 0

        def selector_call(parse_attempt: int) -> str:
            nonlocal client_call_attempts
            for infra_attempt in range(1, MAX_INFRA_ATTEMPTS + 1):
                client_call_attempts += 1
                logical_attempt: int = (
                    (parse_attempt - 1) * MAX_INFRA_ATTEMPTS
                    + infra_attempt
                )
                context: ExecutionContext = ExecutionContext(
                    job_id,
                    served_model,
                    domain,
                    SELECT_ARM,
                    instance_id,
                    logical_attempt,
                    prepared["selector_payload_hash"],
                    prepared["execution_request_hash"],
                )
                started_at: float = time.monotonic()
                try:
                    with bind_execution_context(context):
                        response: str = runtime["chat"](
                            client,
                            served_model,
                            prepared["rendered_prompt"],
                            None,
                            TEMPERATURE,
                            MAX_TOKENS,
                            None,
                            extra_body,
                        )
                    append_jsonl(
                        attempt_log_path,
                        {
                            "schema_version": (
                                "runtime-matched-selector-attempt-v1"
                            ),
                            "job_id": job_id,
                            "model": served_model,
                            "domain": domain,
                            "arm": SELECT_ARM,
                            "instance_id": instance_id,
                            "parse_attempt": parse_attempt,
                            "infra_attempt": infra_attempt,
                            "logical_attempt": logical_attempt,
                            "status": "response",
                            "selector_payload_hash": prepared[
                                "selector_payload_hash"
                            ],
                            "execution_request_hash": prepared[
                                "execution_request_hash"
                            ],
                            "elapsed_seconds": round(
                                time.monotonic() - started_at,
                                6,
                            ),
                            "raw_response": response,
                        },
                        write_lock,
                    )
                    return response
                except runtime["request_error_types"] as error:
                    details = error_context(error)
                    category: FailureCategory = classify_request_error(
                        details.exception_name,
                        details.message,
                        details.status_code,
                        details.response_body,
                    )
                    append_jsonl(
                        attempt_log_path,
                        {
                            "schema_version": (
                                "runtime-matched-selector-attempt-v1"
                            ),
                            "job_id": job_id,
                            "model": served_model,
                            "domain": domain,
                            "arm": SELECT_ARM,
                            "instance_id": instance_id,
                            "parse_attempt": parse_attempt,
                            "infra_attempt": infra_attempt,
                            "logical_attempt": logical_attempt,
                            "status": "error",
                            "failure_category": category,
                            "selector_payload_hash": prepared[
                                "selector_payload_hash"
                            ],
                            "execution_request_hash": prepared[
                                "execution_request_hash"
                            ],
                            "exception_name": details.exception_name,
                            "message": details.message,
                            "status_code": details.status_code,
                            "response_body": details.response_body,
                            "elapsed_seconds": round(
                                time.monotonic() - started_at,
                                6,
                            ),
                        },
                        write_lock,
                    )
                    should_retry: bool = (
                        category == "infra_transient"
                        and infra_attempt < MAX_INFRA_ATTEMPTS
                    )
                    if should_retry:
                        print(
                            canonical_json(
                                {
                                    "level": "warning",
                                    "event": "selector_infra_retry",
                                    "job_id": job_id,
                                    "model": served_model,
                                    "domain": domain,
                                    "instance_id": instance_id,
                                    "parse_attempt": parse_attempt,
                                    "infra_attempt": infra_attempt,
                                    "exception_name": (
                                        details.exception_name
                                    ),
                                    "status_code": details.status_code,
                                }
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(
                            RETRY_DELAYS_SECONDS[infra_attempt - 1]
                        )
                        continue
                    raise SelectorRequestFailure(
                        category,
                        details.exception_name,
                        details.message,
                        details.status_code,
                        details.response_body,
                    ) from error
            raise AssertionError(
                "Selector infrastructure retry loop exited without a result"
            )

        outcome: SelectionOutcome = run_selection_protocol(
            selector_call,
            runtime["parse_first_number"],
            len(prepared["candidates"]),
        )
        return build_selection_record(
            prepared,
            outcome,
            result_tag,
            served_model,
            domain,
            runtime_manifest_sha256,
            code_bundle_sha256,
            candidate_source_sha256,
            client_call_attempts,
        )

    pending_ids: list[str] = sorted(
        set(prepared_index) - set(existing_index)
    )
    selected_ids: list[str] = (
        pending_ids
        if max_new_records == 0
        else pending_ids[:max_new_records]
    )
    pending: list[PreparedSelection] = [
        prepared_index[instance_id] for instance_id in selected_ids
    ]
    if pending:
        futures: dict[Future[SelectionRecord], str] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for prepared in pending:
                instance_id: str = require_string(
                    prepared["instance"].get("instance_id"),
                    "prepared.instance_id",
                )
                futures[executor.submit(run_one, prepared)] = instance_id
            completed: int = 0
            for future in as_completed(futures):
                record: SelectionRecord = future.result()
                append_jsonl(
                    output_path,
                    cast(AttemptPayload, record),
                    write_lock,
                )
                existing_index[record["instance_id"]] = record
                completed += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        canonical_json(
                            {
                                "event": "selector_progress",
                                "job_id": job_id,
                                "model": served_model,
                                "domain": domain,
                                "completed_this_run": completed,
                                "pending_this_run": len(futures),
                                "total_records": len(existing_index),
                            }
                        ),
                        flush=True,
                    )

    final_records: list[SelectionRecord] = read_records(output_path)
    final_ids: list[str] = [
        record["instance_id"] for record in final_records
    ]
    missing_ids: list[str] = sorted(set(instance_index) - set(final_ids))
    unexpected_ids: list[str] = sorted(
        set(final_ids) - set(instance_index)
    )
    duplicate_count: int = len(final_ids) - len(set(final_ids))
    category_counts: dict[str, int] = {}
    for record in final_records:
        category: str = record["failure_category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    unresolved: int = (
        category_counts.get("infra_transient", 0)
        + category_counts.get("unclassified_error", 0)
    )
    summary: JsonObject = {
        "event": "runtime_matched_select_decisions_complete",
        "job_id": job_id,
        "model": served_model,
        "domain": domain,
        "run_mode": "full" if max_new_records == 0 else "canary",
        "expected": expected_count,
        "observed": len(final_records),
        "selected_this_run": len(selected_ids),
        "missing": len(missing_ids),
        "unexpected": len(unexpected_ids),
        "duplicates": duplicate_count,
        "failure_categories": category_counts,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "attempt_log": str(attempt_log_path),
    }
    print(canonical_json(summary), flush=True)
    full_run: bool = max_new_records == 0
    if (
        unexpected_ids
        or duplicate_count
        or unresolved
        or (
            full_run
            and (
                len(final_records) != expected_count
                or missing_ids
            )
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
