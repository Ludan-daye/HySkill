#!/usr/bin/env python3
"""Export routed Select loading decisions before answer inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    FailureCategory,
    JsonObject,
    JsonValue,
    audit_record_coverage,
    canonical_json,
    select_expected_skill_ids,
    sha256_file,
    validate_failure_category,
)
from hyskill.loading_metrics import LoadingDecisionRow
from scripts.audit_k2_reuse import (
    load_decisions,
    load_instances,
    require_list,
    require_object,
    require_string,
)
from scripts.export_k2_loading_decisions import (
    gold_skill_ids,
    load_validation_ids,
    write_rows,
)
from scripts.run_select_only import SelectionRecord, read_selection_records


def parse_args() -> argparse.Namespace:
    """Parse one model-domain Select loading export job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--selected-source", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--taus", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def build_select_row(
    instance_id: str,
    model: str,
    domain: str,
    expected: tuple[str, ...],
    gold: list[str],
    is_validation: bool,
    category: FailureCategory,
    decision_source_sha256: str,
) -> LoadingDecisionRow:
    """Build one persisted selector loading decision."""

    loaded: bool = bool(expected)
    gold_set: set[str] = set(gold)
    hit: bool | None = (
        any(skill_id in gold_set for skill_id in expected) if loaded else None
    )
    return {
        "schema_version": "k2-loading-decision-v1",
        "instance_id": instance_id,
        "model": model,
        "domain": domain,
        "arm": "routed_select",
        "expected_skill_ids": list(expected),
        "gold_skill_ids": gold,
        "loaded": loaded,
        "hit": hit,
        "gold_loaded": hit is True,
        "is_validation": is_validation,
        "failure_category": category,
        "decision_source_sha256": decision_source_sha256,
    }


def main() -> None:
    """Export one complete selection-only decision set."""

    args = parse_args()
    instances_path: Path = cast(Path, args.instances).resolve()
    selected_source_path: Path = cast(Path, args.selected_source).resolve()
    selection_path: Path = cast(Path, args.selection).resolve()
    taus_path: Path = cast(Path, args.taus).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    model: str = str(args.model)
    domain: str = str(args.domain)
    expected_count: int = int(args.expected_count)

    instances: list[JsonObject] = load_instances(instances_path, domain)
    if len(instances) != expected_count:
        raise DownstreamDataError(
            "Select loading denominator mismatch: "
            f"expected={expected_count}, actual={len(instances)}"
        )
    instance_index: dict[str, JsonObject] = {
        require_string(instance.get("instance_id"), "instance.instance_id"): instance
        for instance in instances
    }
    decisions: dict[str, JsonObject] = load_decisions(selected_source_path)
    records: list[SelectionRecord] = read_selection_records(selection_path)
    selection_index: dict[str, SelectionRecord] = {
        record["instance_id"]: record for record in records
    }
    for name, observed_ids in (
        ("selected-source", list(decisions)),
        ("selection", list(selection_index)),
    ):
        coverage = audit_record_coverage(list(instance_index), observed_ids)
        if not coverage.complete:
            raise DownstreamDataError(
                f"{name} coverage mismatch: "
                f"missing={list(coverage.missing_ids)[:20]}, "
                f"unexpected={list(coverage.unexpected_ids)[:20]}"
            )
    validation_ids: frozenset[str] = load_validation_ids(taus_path)
    unknown_validation_ids: list[str] = sorted(validation_ids - set(instance_index))
    if unknown_validation_ids:
        raise DownstreamDataError(
            "Gate taus contains IDs outside Select instance support: "
            f"sample={unknown_validation_ids[:20]}"
        )
    decision_source_sha256: str = sha256_file(selected_source_path)
    rows: list[LoadingDecisionRow] = []
    for instance_id, instance in instance_index.items():
        selection: SelectionRecord = selection_index[instance_id]
        category: FailureCategory = validate_failure_category(
            selection.get("failure_category")
        )
        if category in ("infra_transient", "unclassified_error"):
            raise DownstreamDataError(
                "Select loading export cannot include unresolved failures: "
                f"instance_id={instance_id}, failure_category={category}"
            )
        raw_selected_skill_id: JsonValue | None = selection.get(
            "selected_skill_id"
        )
        selected_skill_id: str | None
        if raw_selected_skill_id is None:
            selected_skill_id = None
        elif isinstance(raw_selected_skill_id, str) and raw_selected_skill_id:
            selected_skill_id = raw_selected_skill_id
        else:
            raise DownstreamDataError(
                "Selection record has invalid selected_skill_id: "
                f"instance_id={instance_id}, value={raw_selected_skill_id!r}"
            )
        expected: tuple[str, ...] = select_expected_skill_ids(
            selected_skill_id,
            category,
        )
        raw_retrieved: list[JsonValue] = require_list(
            decisions[instance_id].get("retrieved"),
            f"selected-source:{instance_id}.retrieved",
        )
        source_skill_ids: tuple[str, ...] = tuple(
            require_string(
                require_object(
                    value,
                    f"selected-source:{instance_id}.retrieved[{index}]",
                ).get("skill_id"),
                f"selected-source:{instance_id}.retrieved[{index}].skill_id",
            )
            for index, value in enumerate(raw_retrieved)
        )
        if source_skill_ids != expected:
            raise DownstreamDataError(
                "Selected source disagrees with selection record: "
                f"instance_id={instance_id}, expected={expected}, "
                f"source={source_skill_ids}"
            )
        rows.append(
            build_select_row(
                instance_id,
                model,
                domain,
                expected,
                gold_skill_ids(instance),
                instance_id in validation_ids,
                category,
                decision_source_sha256,
            )
        )
    write_rows(output_path, rows)
    summary: JsonObject = {
        "event": "k2_select_loading_decisions_exported",
        "model": model,
        "domain": domain,
        "instances": len(instances),
        "records": len(rows),
        "validation_instances": len(validation_ids),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
