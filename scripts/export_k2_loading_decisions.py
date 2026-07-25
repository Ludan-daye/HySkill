#!/usr/bin/env python3
"""Export routed Always and Gated loading decisions before answer inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    JsonObject,
    JsonValue,
    always_expected_skill_ids,
    audit_record_coverage,
    canonical_json,
    gated_expected_skill_ids,
    sha256_file,
)
from hyskill.loading_metrics import LoadingDecisionRow
from scripts.audit_k2_reuse import (
    load_decisions,
    load_instances,
    require_list,
    require_object,
    require_string,
)

DeterministicLoadingArm = Literal["routed_always", "routed_gated"]


def parse_args() -> argparse.Namespace:
    """Parse one model-domain Always/Gated export job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--always-source", required=True, type=Path)
    parser.add_argument("--gated-source", required=True, type=Path)
    parser.add_argument("--taus", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_validation_ids(path: Path) -> frozenset[str]:
    """Load the exact gate calibration IDs from a taus file."""

    if not path.is_file():
        raise FileNotFoundError(f"Gate taus file does not exist: path={path}")
    try:
        raw_payload: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise DownstreamDataError(
            "Gate taus JSON is malformed: "
            f"path={path}, line={error.lineno}, column={error.colno}, "
            f"message={error.msg}"
        ) from error
    payload: JsonObject = require_object(raw_payload, f"taus:{path}")
    raw_ids: list[JsonValue] = require_list(
        payload.get("val_ids"),
        f"taus:{path}.val_ids",
    )
    validation_ids: list[str] = [
        require_string(value, f"taus:{path}.val_ids[{index}]")
        for index, value in enumerate(raw_ids)
    ]
    if len(validation_ids) != len(set(validation_ids)):
        raise DownstreamDataError(
            f"Gate taus contains duplicate validation IDs: path={path}"
        )
    return frozenset(validation_ids)


def gold_skill_ids(instance: JsonObject) -> list[str]:
    """Load the exact ordered gold skill IDs from one instance."""

    instance_id: str = require_string(
        instance.get("instance_id"),
        "instance.instance_id",
    )
    raw_gold: list[JsonValue] = require_list(
        instance.get("skill_annotations"),
        f"instance:{instance_id}.skill_annotations",
    )
    gold: list[str] = [
        require_string(
            skill_id,
            f"instance:{instance_id}.skill_annotations[{index}]",
        )
        for index, skill_id in enumerate(raw_gold)
    ]
    if not gold:
        raise DownstreamDataError(
            f"Loading instance has no gold skills: instance_id={instance_id}"
        )
    return gold


def build_loading_row(
    instance_id: str,
    model: str,
    domain: str,
    arm: DeterministicLoadingArm,
    expected_skill_ids: tuple[str, ...],
    gold: list[str],
    is_validation: bool,
    decision_source_sha256: str,
) -> LoadingDecisionRow:
    """Build one deterministic loading decision record."""

    if arm not in ("routed_always", "routed_gated"):
        raise ValueError(f"Unsupported deterministic loading arm: arm={arm}")
    loaded: bool = bool(expected_skill_ids)
    hit: bool | None = (
        any(skill_id in set(gold) for skill_id in expected_skill_ids)
        if loaded
        else None
    )
    return {
        "schema_version": "k2-loading-decision-v1",
        "instance_id": instance_id,
        "model": model,
        "domain": domain,
        "arm": arm,
        "expected_skill_ids": list(expected_skill_ids),
        "gold_skill_ids": gold,
        "loaded": loaded,
        "hit": hit,
        "gold_loaded": hit is True,
        "is_validation": is_validation,
        "failure_category": "success",
        "decision_source_sha256": decision_source_sha256,
    }


def write_rows(path: Path, rows: list[LoadingDecisionRow]) -> None:
    """Atomically replace one loading-decision JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(canonical_json(row) + "\n")
    temporary_path.replace(path)


def main() -> None:
    """Export two deterministic arms on exact instance support."""

    args = parse_args()
    instances_path: Path = cast(Path, args.instances).resolve()
    always_source_path: Path = cast(Path, args.always_source).resolve()
    gated_source_path: Path = cast(Path, args.gated_source).resolve()
    taus_path: Path = cast(Path, args.taus).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    model: str = str(args.model)
    domain: str = str(args.domain)
    expected_count: int = int(args.expected_count)

    instances: list[JsonObject] = load_instances(instances_path, domain)
    if len(instances) != expected_count:
        raise DownstreamDataError(
            "Loading denominator mismatch: "
            f"expected={expected_count}, actual={len(instances)}"
        )
    instance_index: dict[str, JsonObject] = {
        require_string(instance.get("instance_id"), "instance.instance_id"): instance
        for instance in instances
    }
    always_decisions: dict[str, JsonObject] = load_decisions(always_source_path)
    gated_decisions: dict[str, JsonObject] = load_decisions(gated_source_path)
    for name, decisions in (
        ("always", always_decisions),
        ("gated", gated_decisions),
    ):
        coverage = audit_record_coverage(list(instance_index), list(decisions))
        if not coverage.complete:
            raise DownstreamDataError(
                f"{name} decision coverage mismatch: "
                f"missing={list(coverage.missing_ids)[:20]}, "
                f"unexpected={list(coverage.unexpected_ids)[:20]}"
            )
    validation_ids: frozenset[str] = load_validation_ids(taus_path)
    unknown_validation_ids: list[str] = sorted(validation_ids - set(instance_index))
    if unknown_validation_ids:
        raise DownstreamDataError(
            "Gate taus contains IDs outside the instance set: "
            f"sample={unknown_validation_ids[:20]}"
        )
    rows: list[LoadingDecisionRow] = []
    always_source_sha256: str = sha256_file(always_source_path)
    gated_source_sha256: str = sha256_file(gated_source_path)
    for instance_id, instance in instance_index.items():
        gold: list[str] = gold_skill_ids(instance)
        raw_always: list[JsonValue] = require_list(
            always_decisions[instance_id].get("retrieved"),
            f"always:{instance_id}.retrieved",
        )
        raw_gated: list[JsonValue] = require_list(
            gated_decisions[instance_id].get("retrieved"),
            f"gated:{instance_id}.retrieved",
        )
        always_retrieved: list[JsonObject] = [
            require_object(value, f"always:{instance_id}.retrieved[{index}]")
            for index, value in enumerate(raw_always)
        ]
        gated_retrieved: list[JsonObject] = [
            require_object(value, f"gated:{instance_id}.retrieved[{index}]")
            for index, value in enumerate(raw_gated)
        ]
        always_expected: tuple[str, ...] = always_expected_skill_ids(
            always_retrieved
        )
        gated_expected: tuple[str, ...] = gated_expected_skill_ids(
            gated_retrieved
        )
        if gated_expected and gated_expected != always_expected:
            raise DownstreamDataError(
                "Gated source changed routed top-1 instead of only blocking: "
                f"instance_id={instance_id}, always={always_expected}, "
                f"gated={gated_expected}"
            )
        is_validation: bool = instance_id in validation_ids
        rows.append(
            build_loading_row(
                instance_id,
                model,
                domain,
                "routed_always",
                always_expected,
                gold,
                is_validation,
                always_source_sha256,
            )
        )
        rows.append(
            build_loading_row(
                instance_id,
                model,
                domain,
                "routed_gated",
                gated_expected,
                gold,
                is_validation,
                gated_source_sha256,
            )
        )
    write_rows(output_path, rows)
    summary: JsonObject = {
        "event": "k2_loading_decisions_exported",
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
