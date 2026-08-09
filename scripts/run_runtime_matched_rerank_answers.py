#!/usr/bin/env python3
"""Answer from persisted native-rerank decisions with zero-call failures."""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

from hyskill.runtime_matched_execution import (
    JsonLike,
    OpenAIClientLike,
    canonical_json,
    sha256_file,
    wrap_openai_client,
)
from hyskill.runtime_matched_rerank import (
    FROZEN_SRAGENTS_REVISION,
    RERANK_ANSWER_JOB_ARM,
    RERANK_ANSWER_STAGE,
    RERANK_ANSWER_SCHEMA_VERSION,
    RERANK_ARM,
    RERANK_DECISION_SCHEMA_VERSION,
    RERANK_DECISION_STAGE,
    RETRY_DELAYS_SECONDS,
    AnswerInput,
    DirectEngine,
    JsonObject,
    NativeRerankRuntime,
    RuntimeMatchedRerankError,
    answer_generation,
    append_jsonl,
    assert_exact_coverage,
    build_answer_input,
    build_answer_record,
    build_zero_call_answer_record,
    decision_failure_hashes,
    indexed_rows,
    load_corpus,
    load_instances,
    load_native_rerank_runtime,
    now_sleep,
    read_jsonl,
    require_object,
    require_string,
    require_supported_model,
    run_answer_one,
    validate_job_manifest,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-domain rerank answer job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt-log", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--max-new-records", required=True, type=int)
    parser.add_argument("--sragents-checkout", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument(
        "--sragents-revision",
        required=True,
        choices=(FROZEN_SRAGENTS_REVISION,),
    )
    return parser.parse_args()


def validate_decision(
    decision: JsonObject,
    instance_id: str,
    model_tag: str,
    model: str,
    domain: str,
) -> str:
    """Validate one decision and return its failure category."""

    for field, expected in (
        ("schema_version", RERANK_DECISION_SCHEMA_VERSION),
        ("instance_id", instance_id),
        ("domain", domain),
        ("arm", RERANK_ARM),
        ("stage", RERANK_DECISION_STAGE),
        ("model", model_tag),
        ("served_model", model),
    ):
        if decision.get(field) != expected:
            raise RuntimeMatchedRerankError(
                "Rerank decision identity mismatch: "
                f"instance_id={instance_id}, field={field}, "
                f"expected={expected!r}, actual={decision.get(field)!r}"
            )
    category: str = require_string(
        decision.get("failure_category"),
        f"decision:{instance_id}.failure_category",
    )
    if category not in (
        "success",
        "method_failure",
        "infra_transient",
        "unclassified_error",
    ):
        raise RuntimeMatchedRerankError(
            "Rerank decision has invalid failure category: "
            f"instance_id={instance_id}, category={category}"
        )
    if category == "success":
        selected_skill_id: str = require_string(
            decision.get("selected_skill_id"),
            f"decision:{instance_id}.selected_skill_id",
        )
        raw_reranked = decision.get("reranked_candidate_ids")
        if not isinstance(raw_reranked, list) or len(raw_reranked) != 50:
            raise RuntimeMatchedRerankError(
                "Successful rerank decision must contain 50 ordered IDs: "
                f"instance_id={instance_id}"
            )
        if raw_reranked[0] != selected_skill_id:
            raise RuntimeMatchedRerankError(
                "Selected skill is not reranked top-1: "
                f"instance_id={instance_id}, selected={selected_skill_id}, "
                f"top1={raw_reranked[0]!r}"
            )
    elif decision.get("selected_skill_id") is not None:
        raise RuntimeMatchedRerankError(
            "Failed rerank decision contains a selected skill: "
            f"instance_id={instance_id}, "
            f"selected={decision.get('selected_skill_id')!r}"
        )
    return category


def validate_existing_answer(
    row: JsonObject,
    instance_id: str,
    decision: JsonObject,
    answer_input: AnswerInput | None,
    model_tag: str,
    model: str,
    domain: str,
    runtime_manifest_sha256: str,
    code_bundle_sha256: str,
    decision_source_sha256: str,
    zero_call_payload_hash: str | None,
    zero_call_request_hash: str | None,
) -> None:
    """Reject stale, reused, or cross-job answer rows on resume."""

    expected_payload_hash: str = (
        answer_input["answer_payload_hash"]
        if answer_input is not None
        else require_string(
            cast(JsonLike, zero_call_payload_hash),
            "zero_call_payload_hash",
        )
    )
    expected_request_hash: str = (
        answer_input["execution_request_hash"]
        if answer_input is not None
        else require_string(
            cast(JsonLike, zero_call_request_hash),
            "zero_call_request_hash",
        )
    )
    expected_fields: tuple[tuple[str, object], ...] = (
        ("schema_version", RERANK_ANSWER_SCHEMA_VERSION),
        ("instance_id", instance_id),
        ("domain", domain),
        ("arm", RERANK_ARM),
        ("stage", RERANK_ANSWER_STAGE),
        ("model", model_tag),
        ("served_model", model),
        ("answer_payload_hash", expected_payload_hash),
        ("execution_request_hash", expected_request_hash),
        (
            "decision_execution_request_hash",
            decision.get("execution_request_hash"),
        ),
        ("runtime_manifest_sha256", runtime_manifest_sha256),
        ("code_bundle_sha256", code_bundle_sha256),
        ("decision_source_sha256", decision_source_sha256),
        ("reused_same_arm", False),
    )
    for field, expected in expected_fields:
        if row.get(field) != expected:
            raise RuntimeMatchedRerankError(
                "Existing rerank answer is stale: "
                f"instance_id={instance_id}, field={field}, "
                f"expected={expected!r}, actual={row.get(field)!r}"
            )
    decision_category: object = decision.get("failure_category")
    answer_category: object = row.get("failure_category")
    if decision_category == "method_failure":
        if (
            row.get("zero_call") is not True
            or row.get("engine_attempts") != 0
            or answer_category != "method_failure"
            or row.get("skill_ids_used") != []
            or row.get("expected_skill_ids") != []
        ):
            raise RuntimeMatchedRerankError(
                "Decision method failure does not map to an exact zero-call "
                "answer: "
                f"instance_id={instance_id}, row={row!r}"
            )
        return
    selected_skill_id: str = require_string(
        decision.get("selected_skill_id"),
        f"decision:{instance_id}.selected_skill_id",
    )
    if (
        row.get("zero_call") is not False
        or row.get("expected_skill_ids") != [selected_skill_id]
    ):
        raise RuntimeMatchedRerankError(
            "Answer row does not preserve the successful rerank decision: "
            f"instance_id={instance_id}, selected_skill_id={selected_skill_id}, "
            f"zero_call={row.get('zero_call')!r}, "
            f"expected_skill_ids={row.get('expected_skill_ids')!r}"
        )
    if answer_category == "success":
        if (
            row.get("skill_ids_used") != [selected_skill_id]
            or not isinstance(row.get("raw_output"), str)
            or not cast(str, row.get("raw_output")).strip()
        ):
            raise RuntimeMatchedRerankError(
                "Successful answer does not confirm the selected skill and "
                "non-empty output: "
                f"instance_id={instance_id}, "
                f"skill_ids_used={row.get('skill_ids_used')!r}"
            )


def main() -> None:
    """Execute successful decisions and materialize decision failures."""

    args = parse_args()
    workers: int = int(args.workers)
    max_new_records: int = int(args.max_new_records)
    if workers <= 0:
        raise ValueError(f"workers must be positive: workers={workers}")
    if max_new_records < 0:
        raise ValueError(
            "max-new-records must be zero or positive: "
            f"value={max_new_records}"
        )
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    decisions_path: Path = cast(Path, args.decisions).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    attempt_log_path: Path = cast(Path, args.attempt_log).resolve()
    manifest_path: Path = cast(Path, args.runtime_manifest).resolve()
    checkout_path: Path = cast(Path, args.sragents_checkout).resolve()
    repository_root: Path = cast(Path, args.repository_root).resolve()
    model_tag: str = str(args.result_tag)
    model: str = str(args.model)
    api_base: str = str(args.api_base)
    domain: str = str(args.domain)
    revision: str = str(args.sragents_revision)
    require_supported_model(model_tag)

    runtime: NativeRerankRuntime = load_native_rerank_runtime(
        checkout_path,
        revision,
    )
    extra_body: JsonObject | None = runtime["get_extra_body"](
        model,
        False,
    )
    generation: JsonObject = answer_generation(extra_body)
    manifest = validate_job_manifest(
        manifest_path,
        model_tag,
        model,
        api_base,
        domain,
        RERANK_ANSWER_JOB_ARM,
        RERANK_ANSWER_STAGE,
        generation,
        (
            ("corpus", corpus_path),
            ("rerank_decisions", decisions_path),
            ("instances", instances_path),
        ),
        (
            "external/SR-Agents/src/sragents/infer/base.py",
            "external/SR-Agents/src/sragents/infer/engines/direct.py",
            "external/SR-Agents/src/sragents/infer/engines/tool_loop.py",
            "external/SR-Agents/src/sragents/llm.py",
            "external/SR-Agents/src/sragents/prompts.py",
            "hyskill/runtime_matched_execution.py",
            "hyskill/runtime_matched_rerank.py",
            "scripts/run_runtime_matched_rerank_answers.py",
        ),
        repository_root,
    )
    runtime_manifest_sha256: str = sha256_file(manifest_path)
    decision_source_sha256: str = sha256_file(decisions_path)
    instances: list[JsonObject] = load_instances(instances_path, domain)
    instance_index: dict[str, JsonObject] = indexed_rows(
        instances,
        "instances",
    )
    instance_ids: list[str] = list(instance_index)
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    decisions: list[JsonObject] = read_jsonl(decisions_path)
    decision_index: dict[str, JsonObject] = indexed_rows(
        decisions,
        "decisions",
    )
    assert_exact_coverage(
        instance_ids,
        list(decision_index),
        "rerank-decisions",
    )
    categories: dict[str, str] = {}
    unresolved_decisions: list[str] = []
    answer_inputs: dict[str, AnswerInput] = {}
    zero_call_hashes: dict[str, tuple[str, str]] = {}
    for instance_id, instance in instance_index.items():
        decision: JsonObject = decision_index[instance_id]
        category: str = validate_decision(
            decision,
            instance_id,
            model_tag,
            model,
            domain,
        )
        categories[instance_id] = category
        if category == "success":
            selected_skill_id: str = require_string(
                decision.get("selected_skill_id"),
                f"decision:{instance_id}.selected_skill_id",
            )
            skill: JsonObject | None = corpus.get(selected_skill_id)
            if skill is None:
                raise RuntimeMatchedRerankError(
                    "Selected skill is absent from corpus: "
                    f"instance_id={instance_id}, skill_id={selected_skill_id}"
                )
            answer_inputs[instance_id] = build_answer_input(
                instance,
                skill,
                runtime,
                manifest,
                runtime_manifest_sha256,
            )
        elif category == "method_failure":
            zero_call_hashes[instance_id] = decision_failure_hashes(
                instance,
                decision,
                manifest,
                runtime_manifest_sha256,
            )
        else:
            unresolved_decisions.append(instance_id)
    if unresolved_decisions:
        raise RuntimeMatchedRerankError(
            "Answering is blocked by unresolved decision failures: "
            f"count={len(unresolved_decisions)}, "
            f"sample={unresolved_decisions[:20]}"
        )

    existing_rows: list[JsonObject] = read_jsonl(output_path)
    retained_rows: list[JsonObject] = [
        row
        for row in existing_rows
        if row.get("failure_category") != "infra_transient"
    ]
    if len(retained_rows) != len(existing_rows):
        write_jsonl_atomic(output_path, retained_rows)
    existing_index: dict[str, JsonObject] = indexed_rows(
        retained_rows,
        "rerank-answer-output",
    )
    unexpected_existing: list[str] = sorted(
        set(existing_index) - set(instance_index)
    )
    if unexpected_existing:
        raise RuntimeMatchedRerankError(
            "Answer output contains unexpected instances: "
            f"sample={unexpected_existing[:20]}"
        )
    for instance_id, row in existing_index.items():
        zero_hashes: tuple[str, str] | None = zero_call_hashes.get(
            instance_id
        )
        validate_existing_answer(
            row,
            instance_id,
            decision_index[instance_id],
            answer_inputs.get(instance_id),
            model_tag,
            model,
            domain,
            runtime_manifest_sha256,
            manifest["code_bundle_sha256"],
            decision_source_sha256,
            zero_hashes[0] if zero_hashes is not None else None,
            zero_hashes[1] if zero_hashes is not None else None,
        )

    pending_all_ids: list[str] = sorted(
        set(instance_ids) - set(existing_index)
    )
    selected_ids: list[str] = (
        pending_all_ids
        if max_new_records == 0
        else pending_all_ids[:max_new_records]
    )
    write_lock: threading.Lock = threading.Lock()
    for instance_id in selected_ids:
        if instance_id not in zero_call_hashes:
            continue
        if instance_id in existing_index:
            continue
        record: JsonObject = build_zero_call_answer_record(
            instance_index[instance_id],
            decision_index[instance_id],
            model_tag,
            model,
            domain,
            manifest,
            runtime_manifest_sha256,
            decision_source_sha256,
        )
        append_jsonl(output_path, record, write_lock)
        existing_index[instance_id] = record

    def usage_sink(event: Mapping[str, JsonLike]) -> None:
        append_jsonl(attempt_log_path, event, write_lock)

    raw_client: OpenAIClientLike = runtime["create_client"](api_base, None)
    client: OpenAIClientLike = wrap_openai_client(raw_client, usage_sink)
    engine: DirectEngine = runtime["create_engine"](
        temperature=0.7,
        max_tokens=2048,
        thinking=False,
    )
    job: JsonObject = require_object(
        manifest["runtime_facts"].get("job"),
        "runtime_facts.job",
    )
    job_id: str = require_string(job.get("job_id"), "job.job_id")
    inference_ids: list[str] = [
        instance_id
        for instance_id in selected_ids
        if categories[instance_id] == "success"
        and instance_id not in existing_index
    ]

    def run_one(instance_id: str) -> JsonObject:
        answer_input: AnswerInput = answer_inputs[instance_id]
        decision: JsonObject = decision_index[instance_id]
        outcome = run_answer_one(
            answer_input,
            runtime,
            engine,
            client,
            model,
            domain,
            job_id,
            now_sleep,
            RETRY_DELAYS_SECONDS,
        )
        selected_skill_id: str = require_string(
            decision.get("selected_skill_id"),
            f"decision:{instance_id}.selected_skill_id",
        )
        if (
            outcome["failure_category"] == "success"
            and outcome["skill_ids_used"] != [selected_skill_id]
        ):
            raise RuntimeMatchedRerankError(
                "Direct engine did not confirm the selected skill: "
                f"instance_id={instance_id}, expected={[selected_skill_id]}, "
                f"actual={outcome['skill_ids_used']}"
            )
        return build_answer_record(
            answer_input,
            decision,
            outcome,
            model_tag,
            model,
            domain,
            runtime_manifest_sha256,
            manifest["code_bundle_sha256"],
            decision_source_sha256,
        )

    futures: dict[Future[JsonObject], str] = {}
    if inference_ids:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for instance_id in inference_ids:
                futures[executor.submit(run_one, instance_id)] = instance_id
            completed: int = 0
            for future in as_completed(futures):
                record: JsonObject = future.result()
                append_jsonl(output_path, record, write_lock)
                existing_index[
                    require_string(
                        record.get("instance_id"),
                        "answer.instance_id",
                    )
                ] = record
                completed += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        canonical_json(
                            {
                                "event": "rerank_answer_progress",
                                "result_tag": model_tag,
                                "served_model": model,
                                "domain": domain,
                                "completed_this_run": completed,
                                "pending_this_run": len(futures),
                            }
                        ),
                        flush=True,
                    )

    final_rows: list[JsonObject] = read_jsonl(output_path)
    final_ids: list[str] = [
        require_string(row.get("instance_id"), "answer.instance_id")
        for row in final_rows
    ]
    unexpected_final: list[str] = sorted(set(final_ids) - set(instance_ids))
    if unexpected_final:
        raise RuntimeMatchedRerankError(
            "Answer output contains unexpected final instances: "
            f"sample={unexpected_final[:20]}"
        )
    missing_after_run: int = len(set(instance_ids) - set(final_ids))
    category_counts: dict[str, int] = {}
    zero_call_count: int = 0
    for row in final_rows:
        category: str = require_string(
            row.get("failure_category"),
            "answer.failure_category",
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        if row.get("zero_call") is True:
            zero_call_count += 1
    print(
        canonical_json(
            {
                "event": "rerank_answer_complete",
                "result_tag": model_tag,
                "served_model": model,
                "domain": domain,
                "run_mode": (
                    "full" if max_new_records == 0 else "canary"
                ),
                "expected": len(instance_ids),
                "observed": len(final_rows),
                "selected_this_run": len(selected_ids),
                "missing_after_run": missing_after_run,
                "zero_call_method_failures": zero_call_count,
                "reused_same_arm": 0,
                "failure_categories": category_counts,
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "attempt_log": str(attempt_log_path),
            }
        ),
        flush=True,
    )
    unresolved: int = (
        category_counts.get("infra_transient", 0)
        + category_counts.get("unclassified_error", 0)
    )
    full_run: bool = max_new_records == 0
    if unresolved or (full_run and missing_after_run):
        print(
            "Rerank answer job is incomplete: "
            f"unresolved={unresolved}, missing={missing_after_run}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
