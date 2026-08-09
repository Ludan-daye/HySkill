"""Focused tests for actual runtime-matched usage aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from scripts.summarize_runtime_matched_usage import (
    DOMAINS,
    JsonObject,
    assert_globally_unique_calls,
    load_usage_events,
    summarize_events,
    usage_log_specs,
)


def usage_event(domain: str, arm: str, index: int) -> JsonObject:
    """Return one attributed response with internally consistent usage."""

    return {
        "schema_version": "runtime-matched-usage-event-v1",
        "job_id": f"model-a-{domain}-{arm}-{index}",
        "model": "served-model-a",
        "domain": domain,
        "arm": arm,
        "instance_id": f"{domain}_{index:05d}",
        "logical_attempt": 1,
        "http_subcall": 1,
        "status": "response",
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "total_tokens": 13,
        "usage_missing_reason": None,
        "elapsed_seconds": 0.25,
    }


def write_expected_logs(root: Path) -> None:
    """Write the exact one-model eligible inventory."""

    specs = usage_log_specs(root, ("model-a",), ("model-a",))
    for index, spec in enumerate(specs):
        spec["path"].parent.mkdir(parents=True, exist_ok=True)
        event: JsonObject = usage_event(
            spec["domain"],
            spec["arm"],
            index,
        )
        spec["path"].write_text(
            json.dumps(event) + "\n",
            encoding="utf-8",
        )


def test_usage_inventory_and_aggregation(tmp_path: Path) -> None:
    """Aggregate the frozen Bare, Rerank, and Select job layout."""

    write_expected_logs(tmp_path)
    specs = usage_log_specs(tmp_path, ("model-a",), ("model-a",))
    assert len(specs) == len(DOMAINS) * 5
    events = [
        event
        for spec in specs
        for event in load_usage_events(spec)
    ]
    assert_globally_unique_calls(
        [(spec, load_usage_events(spec)) for spec in specs]
    )
    summary: JsonObject = summarize_events(events)
    assert summary["http_calls"] == 20
    assert summary["usage_missing_calls"] == 0
    assert summary["prompt_tokens"] == 220
    assert summary["completion_tokens"] == 40
    assert summary["total_tokens"] == 260
    assert cast(float, summary["elapsed_seconds_median"]) == 0.25


def test_usage_inventory_uses_frozen_select_file_labels(
    tmp_path: Path,
) -> None:
    """Resolve the known Llama and shared Wave-C filename conventions."""

    llama_specs = usage_log_specs(
        tmp_path,
        ("llama31-8b",),
        ("llama31-8b",),
    )
    qwen_specs = usage_log_specs(
        tmp_path,
        ("qwen35-9b",),
        ("qwen35-9b",),
    )
    llama_select_paths: list[str] = [
        spec["path"].name
        for spec in llama_specs
        if spec["arm"] == "select_bm25"
    ]
    qwen_select_paths: list[str] = [
        spec["path"].name
        for spec in qwen_specs
        if spec["arm"] == "select_bm25"
    ]
    assert llama_select_paths
    assert qwen_select_paths
    assert all("select_bm25" in name for name in llama_select_paths)
    assert all("select-bm25" in name for name in qwen_select_paths)
