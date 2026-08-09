#!/usr/bin/env python3
"""Run one resumable native listwise-rerank decision job."""

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
    RERANK_DECISION_JOB_ARM,
    RERANK_DECISION_STAGE,
    RETRY_DELAYS_SECONDS,
    JsonObject,
    NativeRerankRuntime,
    RerankDecision,
    RerankInput,
    RuntimeMatchedRerankError,
    append_jsonl,
    assert_exact_coverage,
    build_rerank_decision,
    build_rerank_inputs,
    indexed_rows,
    load_corpus,
    load_instances,
    load_native_rerank_runtime,
    load_retrieval_records,
    now_sleep,
    read_jsonl,
    require_object,
    require_string,
    require_supported_model,
    rerank_generation,
    rerank_one,
    validate_existing_decision,
    validate_job_manifest,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    """Parse one explicit model-domain rerank decision job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--bm25-source", required=True, type=Path)
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


def main() -> None:
    """Execute and persist every native rerank decision exactly once."""

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
    source_path: Path = cast(Path, args.bm25_source).resolve()
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
    generation: JsonObject = rerank_generation(extra_body)
    manifest = validate_job_manifest(
        manifest_path,
        model_tag,
        model,
        api_base,
        domain,
        RERANK_DECISION_JOB_ARM,
        RERANK_DECISION_STAGE,
        generation,
        (
            ("bm25_candidates", source_path),
            ("corpus", corpus_path),
            ("instances", instances_path),
        ),
        (
            "external/SR-Agents/src/sragents/corpus.py",
            "external/SR-Agents/src/sragents/llm.py",
            "external/SR-Agents/src/sragents/prompts.py",
            "external/SR-Agents/src/sragents/retrieve/llm_rerank.py",
            "hyskill/runtime_matched_execution.py",
            "hyskill/runtime_matched_rerank.py",
            "scripts/run_runtime_matched_rerank_decisions.py",
        ),
        repository_root,
    )
    runtime_manifest_sha256: str = sha256_file(manifest_path)
    source_sha256: str = sha256_file(source_path)
    instances: list[JsonObject] = load_instances(instances_path, domain)
    instance_ids: list[str] = [
        require_string(instance.get("instance_id"), "instance.instance_id")
        for instance in instances
    ]
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    _source_payload, source_rows = load_retrieval_records(source_path)
    source_index: dict[str, JsonObject] = indexed_rows(
        source_rows,
        "bm25-source",
    )
    assert_exact_coverage(
        instance_ids,
        list(source_index),
        "bm25-source",
    )
    rerank_inputs: list[RerankInput] = build_rerank_inputs(
        instances,
        source_index,
        corpus,
        runtime,
        manifest,
        runtime_manifest_sha256,
    )
    input_index: dict[str, RerankInput] = {
        require_string(
            rerank_input["instance"].get("instance_id"),
            "rerank-input.instance_id",
        ): rerank_input
        for rerank_input in rerank_inputs
    }

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
        "rerank-output",
    )
    unexpected_existing: list[str] = sorted(
        set(existing_index) - set(input_index)
    )
    if unexpected_existing:
        raise RuntimeMatchedRerankError(
            "Rerank output contains unexpected instances: "
            f"sample={unexpected_existing[:20]}"
        )
    for instance_id, row in existing_index.items():
        validate_existing_decision(
            row,
            input_index[instance_id],
            model_tag,
            model,
            domain,
            runtime_manifest_sha256,
            manifest["code_bundle_sha256"],
            source_sha256,
        )

    write_lock: threading.Lock = threading.Lock()

    def usage_sink(event: Mapping[str, JsonLike]) -> None:
        append_jsonl(attempt_log_path, event, write_lock)

    raw_client: OpenAIClientLike = runtime["create_client"](api_base, None)
    client: OpenAIClientLike = wrap_openai_client(raw_client, usage_sink)
    job: JsonObject = require_object(
        manifest["runtime_facts"].get("job"),
        "runtime_facts.job",
    )
    job_id: str = require_string(job.get("job_id"), "job.job_id")
    pending_ids: list[str] = sorted(set(input_index) - set(existing_index))
    selected_ids: list[str] = (
        pending_ids
        if max_new_records == 0
        else pending_ids[:max_new_records]
    )
    selected_inputs: list[RerankInput] = [
        input_index[instance_id] for instance_id in selected_ids
    ]

    def run_one(rerank_input: RerankInput) -> RerankDecision:
        outcome = rerank_one(
            rerank_input,
            runtime,
            client,
            model,
            model_tag,
            domain,
            job_id,
            extra_body,
            now_sleep,
            RETRY_DELAYS_SECONDS,
        )
        return build_rerank_decision(
            rerank_input,
            outcome,
            model_tag,
            model,
            domain,
            runtime_manifest_sha256,
            manifest["code_bundle_sha256"],
            source_sha256,
        )

    futures: dict[Future[RerankDecision], str] = {}
    if selected_inputs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for rerank_input in selected_inputs:
                instance_id: str = require_string(
                    rerank_input["instance"].get("instance_id"),
                    "rerank-input.instance_id",
                )
                futures[executor.submit(run_one, rerank_input)] = instance_id
            completed: int = 0
            for future in as_completed(futures):
                record: RerankDecision = future.result()
                append_jsonl(
                    output_path,
                    cast(Mapping[str, JsonLike], record),
                    write_lock,
                )
                existing_index[record["instance_id"]] = cast(
                    JsonObject,
                    record,
                )
                completed += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        canonical_json(
                            {
                                "event": "rerank_decision_progress",
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
        require_string(row.get("instance_id"), "decision.instance_id")
        for row in final_rows
    ]
    unexpected_final: list[str] = sorted(set(final_ids) - set(instance_ids))
    if unexpected_final:
        raise RuntimeMatchedRerankError(
            "Rerank output contains unexpected final instances: "
            f"sample={unexpected_final[:20]}"
        )
    missing_after_run: int = len(set(instance_ids) - set(final_ids))
    category_counts: dict[str, int] = {}
    for row in final_rows:
        category: str = require_string(
            row.get("failure_category"),
            "decision.failure_category",
        )
        category_counts[category] = category_counts.get(category, 0) + 1
    print(
        canonical_json(
            {
                "event": "rerank_decision_complete",
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
            "Rerank decision job is incomplete: "
            f"unresolved={unresolved}, missing={missing_after_run}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
