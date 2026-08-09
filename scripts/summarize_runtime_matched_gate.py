#!/usr/bin/env python3
"""Validate and combine the exact 32 runtime-matched Gate audits."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, TypedDict, cast

from hyskill.runtime_matched_execution import (
    FROZEN_K2_RUNTIME_REFERENCES,
    JsonValue,
    canonical_json,
    require_sha256,
    sha256_file,
    sha256_json,
)
from hyskill.runtime_matched_gate import (
    GATE_AUDIT_SCHEMA_VERSION,
    GATE_COMBINED_RERUN_SCHEMA_VERSION,
    GATE_DECISION_SCHEMA_VERSION,
    GATE_DIFF_ROW_SCHEMA_VERSION,
    GATE_RERUN_ANSWER_SCHEMA_VERSION,
    GATE_RERUN_MANIFEST_SCHEMA_VERSION,
    P_MIN,
    RULE_DOMAIN_COUNTS,
    GateArm,
    GateTaskKey,
    JsonObject,
    RuntimeMatchedGateError,
    expected_gate_row_count,
    expected_gate_task_keys,
    require_boolean,
    require_gate_arm,
    require_list,
    require_number,
    require_object,
    require_optional_number,
    require_string,
    require_string_list,
)


COMBINED_DIFF_SCHEMA_VERSION: str = "runtime-matched-gate-combined-diff-v1"
GateRowKey: TypeAlias = tuple[str, str, GateArm, str]


class GateTaskArtifacts(TypedDict):
    """Validated artifacts for one model-domain-arm Gate task."""

    key: GateTaskKey
    model: str
    served_model: str
    domain: str
    arm: GateArm
    expected_count: int
    audit_path: Path
    audit_sha256: str
    audit: JsonObject
    diff_path: Path
    diff_sha256: str
    diff_rows: list[JsonObject]
    decision_path: Path
    decision_sha256: str
    rerun_path: Path
    rerun_sha256: str
    rerun_rows: list[JsonObject]


def parse_args() -> argparse.Namespace:
    """Parse the exact fleet Gate aggregation inputs and outputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--audits", required=True, nargs="+", type=Path)
    parser.add_argument("--output-diff", required=True, type=Path)
    parser.add_argument("--output-rerun", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path, context: str) -> JsonObject:
    """Load one strict UTF-8 JSON object."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    try:
        raw_value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise RuntimeMatchedGateError(
            f"{context} JSON is malformed: path={path}, "
            f"line={error.lineno}, column={error.colno}, message={error.msg}"
        ) from error
    return require_object(raw_value, f"{context}:{path}")


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load one non-empty strict UTF-8 JSONL artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} file does not exist: path={path}")
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise RuntimeMatchedGateError(
                    f"{context} contains a blank line: "
                    f"path={path}, line={line_number}"
                )
            try:
                raw_value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeMatchedGateError(
                    f"{context} JSONL is malformed: path={path}, "
                    f"line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            rows.append(
                require_object(
                    raw_value,
                    f"{context}:{path}:{line_number}",
                )
            )
    if not rows:
        raise RuntimeMatchedGateError(f"{context} is empty: path={path}")
    return rows


def write_json_atomic(path: Path, payload: Mapping[str, JsonValue]) -> None:
    """Atomically write one formatted JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, JsonValue]],
) -> None:
    """Atomically write canonical JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        for row in rows:
            output_file.write(canonical_json(row))
            output_file.write("\n")
    temporary_path.replace(path)


def require_integer(
    value: JsonValue | None,
    context: str,
) -> int:
    """Return one non-negative JSON integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeMatchedGateError(
            f"Expected non-negative integer: context={context}, value={value!r}"
        )
    return value


def resolve_artifact_path(
    raw_path: str,
    owner_path: Path,
) -> Path:
    """Resolve one recorded artifact path against its owning JSON file."""

    artifact_path: Path = Path(raw_path)
    if not artifact_path.is_absolute():
        artifact_path = owner_path.parent / artifact_path
    return artifact_path.resolve()


def verify_artifact(
    artifact_value: JsonValue | None,
    owner_path: Path,
    context: str,
    hash_cache: dict[Path, str],
) -> tuple[Path, str]:
    """Verify one recorded path and SHA against current bytes."""

    artifact: JsonObject = require_object(artifact_value, context)
    raw_path: str = require_string(artifact.get("path"), f"{context}.path")
    expected_sha256: str = require_sha256(
        artifact.get("sha256"),
        f"{context}.sha256",
    )
    artifact_path: Path = resolve_artifact_path(raw_path, owner_path)
    observed_sha256: str | None = hash_cache.get(artifact_path)
    if observed_sha256 is None:
        observed_sha256 = sha256_file(artifact_path)
        hash_cache[artifact_path] = observed_sha256
    if observed_sha256 != expected_sha256:
        raise RuntimeMatchedGateError(
            f"{context} SHA mismatch: path={artifact_path}, "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    return artifact_path, observed_sha256


def verify_artifact_map(
    raw_map: JsonValue | None,
    owner_path: Path,
    context: str,
    hash_cache: dict[Path, str],
) -> dict[str, tuple[Path, str]]:
    """Verify every named artifact in one audit map."""

    artifact_map: JsonObject = require_object(raw_map, context)
    if not artifact_map:
        raise RuntimeMatchedGateError(f"{context} must not be empty")
    return {
        name: verify_artifact(
            value,
            owner_path,
            f"{context}.{name}",
            hash_cache,
        )
        for name, value in artifact_map.items()
    }


def require_expected_task_matrix(
    task_keys: Sequence[GateTaskKey],
) -> None:
    """Require exactly the frozen 28 routed plus four fixed Gate tasks."""

    expected: tuple[GateTaskKey, ...] = expected_gate_task_keys()
    observed_set: set[GateTaskKey] = set(task_keys)
    expected_set: set[GateTaskKey] = set(expected)
    duplicates: int = len(task_keys) - len(observed_set)
    missing: list[GateTaskKey] = sorted(expected_set - observed_set)
    unexpected: list[GateTaskKey] = sorted(observed_set - expected_set)
    if duplicates or missing or unexpected or len(task_keys) != len(expected):
        raise RuntimeMatchedGateError(
            "Gate audit task matrix mismatch: "
            f"expected={len(expected)}, observed={len(task_keys)}, "
            f"duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )


def task_key_from_audit(
    audit: Mapping[str, JsonValue],
    context: str,
) -> tuple[GateTaskKey, str]:
    """Return and validate one task identity plus served model."""

    model: str = require_string(audit.get("model"), f"{context}.model")
    served_model: str = require_string(
        audit.get("served_model"),
        f"{context}.served_model",
    )
    domain: str = require_string(audit.get("domain"), f"{context}.domain")
    arm: GateArm = require_gate_arm(
        require_string(audit.get("arm"), f"{context}.arm")
    )
    reference = FROZEN_K2_RUNTIME_REFERENCES.get(model)
    if reference is None:
        raise RuntimeMatchedGateError(
            f"Gate audit uses an unknown model: context={context}, model={model}"
        )
    if served_model != reference["served_model"]:
        raise RuntimeMatchedGateError(
            "Gate audit served model mismatch: "
            f"context={context}, expected={reference['served_model']}, "
            f"observed={served_model}"
        )
    return (model, domain, arm), served_model


def _category_counts(
    rows: Sequence[Mapping[str, JsonValue]],
    field_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        value: str = require_string(
            row.get(field_name),
            f"diff[{index}].{field_name}",
        )
        counts[value] = counts.get(value, 0) + 1
    return counts


def _require_thresholds(
    value: JsonValue | None,
    context: str,
) -> JsonObject:
    thresholds: JsonObject = require_object(value, context)
    return {
        "tau1": require_optional_number(
            thresholds.get("tau1"),
            f"{context}.tau1",
        ),
        "tau2": require_optional_number(
            thresholds.get("tau2"),
            f"{context}.tau2",
        ),
    }


def validate_diff_rows(
    rows: Sequence[JsonObject],
    key: GateTaskKey,
    served_model: str,
    expected_count: int,
    old_thresholds: Mapping[str, JsonValue],
    new_thresholds: Mapping[str, JsonValue],
) -> dict[str, JsonObject]:
    """Validate every row-level decision, injection, and payload invariant."""

    model, domain, arm = key
    if len(rows) != expected_count:
        raise RuntimeMatchedGateError(
            "Gate diff row count mismatch: "
            f"key={key}, expected={expected_count}, observed={len(rows)}"
        )
    output: dict[str, JsonObject] = {}
    allowed_decisions: frozenset[str] = frozenset(
        {"blocked_s1", "skipped_s2", "kept"}
    )
    for index, row in enumerate(rows):
        context: str = f"diff:{key}:{index}"
        expected_identity: tuple[tuple[str, JsonValue], ...] = (
            ("schema_version", GATE_DIFF_ROW_SCHEMA_VERSION),
            ("model", model),
            ("served_model", served_model),
            ("domain", domain),
            ("arm", arm),
            ("old_tau1", old_thresholds["tau1"]),
            ("old_tau2", old_thresholds["tau2"]),
            ("new_tau1", new_thresholds["tau1"]),
            ("new_tau2", new_thresholds["tau2"]),
        )
        mismatches: list[str] = [
            f"{field}:expected={expected!r},observed={row.get(field)!r}"
            for field, expected in expected_identity
            if row.get(field) != expected
        ]
        if mismatches:
            raise RuntimeMatchedGateError(
                f"Gate diff identity mismatch: context={context}, "
                f"mismatches={mismatches}"
            )
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        )
        if instance_id in output:
            raise RuntimeMatchedGateError(
                f"Gate diff contains duplicate instance: "
                f"key={key}, instance_id={instance_id}"
            )
        require_boolean(row.get("is_validation"), f"{context}.is_validation")
        require_number(row.get("S1"), f"{context}.S1")
        s2: float = require_number(row.get("S2"), f"{context}.S2")
        if s2 < 0.0 or s2 > 1.0:
            raise RuntimeMatchedGateError(
                f"Gate diff S2 is outside [0, 1]: "
                f"instance_id={instance_id}, value={s2}"
            )
        top1_skill_id: str = require_string(
            row.get("top1_skill_id"),
            f"{context}.top1_skill_id",
        )
        old_decision: str = require_string(
            row.get("old_decision"),
            f"{context}.old_decision",
        )
        new_decision: str = require_string(
            row.get("new_decision"),
            f"{context}.new_decision",
        )
        if (
            old_decision not in allowed_decisions
            or new_decision not in allowed_decisions
        ):
            raise RuntimeMatchedGateError(
                "Gate diff contains an unknown decision: "
                f"instance_id={instance_id}, old={old_decision}, "
                f"new={new_decision}"
            )
        old_skills: list[str] = require_string_list(
            row.get("old_expected_skill_ids"),
            f"{context}.old_expected_skill_ids",
        )
        new_skills: list[str] = require_string_list(
            row.get("new_expected_skill_ids"),
            f"{context}.new_expected_skill_ids",
        )
        for decision, skill_ids, label in (
            (old_decision, old_skills, "old"),
            (new_decision, new_skills, "new"),
        ):
            expected_skills: list[str] = (
                [top1_skill_id] if decision == "kept" else []
            )
            if skill_ids != expected_skills:
                raise RuntimeMatchedGateError(
                    "Gate diff decision and injection disagree: "
                    f"instance_id={instance_id}, side={label}, "
                    f"decision={decision}, expected={expected_skills}, "
                    f"observed={skill_ids}"
                )
        old_payload_hash: str = require_sha256(
            row.get("old_answer_payload_hash"),
            f"{context}.old_answer_payload_hash",
        )
        new_payload_hash: str = require_sha256(
            row.get("new_answer_payload_hash"),
            f"{context}.new_answer_payload_hash",
        )
        require_sha256(
            row.get("old_request_hash"),
            f"{context}.old_request_hash",
        )
        old_failure_category: str = require_string(
            row.get("old_failure_category"),
            f"{context}.old_failure_category",
        )
        if old_failure_category not in ("success", "method_failure"):
            raise RuntimeMatchedGateError(
                "Gate diff contains unresolved old answer outcome: "
                f"instance_id={instance_id}, category={old_failure_category}"
            )
        decision_changed: bool = require_boolean(
            row.get("decision_changed"),
            f"{context}.decision_changed",
        )
        injection_changed: bool = require_boolean(
            row.get("injection_changed"),
            f"{context}.injection_changed",
        )
        payload_changed: bool = require_boolean(
            row.get("payload_changed"),
            f"{context}.payload_changed",
        )
        rerun_required: bool = require_boolean(
            row.get("rerun_required"),
            f"{context}.rerun_required",
        )
        preserve_old_row: bool = require_boolean(
            row.get("preserve_old_row"),
            f"{context}.preserve_old_row",
        )
        observed_relationships: tuple[tuple[str, bool, bool], ...] = (
            ("decision_changed", decision_changed, old_decision != new_decision),
            ("injection_changed", injection_changed, old_skills != new_skills),
            (
                "payload_changed",
                payload_changed,
                old_payload_hash != new_payload_hash,
            ),
            ("rerun_required", rerun_required, payload_changed),
            ("preserve_old_row", preserve_old_row, not payload_changed),
            (
                "injection_payload_equivalence",
                injection_changed,
                payload_changed,
            ),
        )
        relationship_errors: list[str] = [
            f"{name}:observed={observed},expected={expected}"
            for name, observed, expected in observed_relationships
            if observed != expected
        ]
        if relationship_errors:
            raise RuntimeMatchedGateError(
                "Gate diff Boolean invariant mismatch: "
                f"instance_id={instance_id}, errors={relationship_errors}"
            )
        output[instance_id] = row
    return output


def validate_decision_rows(
    rows: Sequence[JsonObject],
    diff_by_id: Mapping[str, JsonObject],
    key: GateTaskKey,
    served_model: str,
) -> None:
    """Require exact one-to-one agreement between decisions and diff rows."""

    if len(rows) != len(diff_by_id):
        raise RuntimeMatchedGateError(
            "Gate decision row count mismatch: "
            f"key={key}, expected={len(diff_by_id)}, observed={len(rows)}"
        )
    seen_ids: set[str] = set()
    model, domain, arm = key
    for index, row in enumerate(rows):
        context: str = f"decision:{key}:{index}"
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        )
        if instance_id in seen_ids:
            raise RuntimeMatchedGateError(
                f"Gate decisions contain duplicate instance: "
                f"key={key}, instance_id={instance_id}"
            )
        seen_ids.add(instance_id)
        diff: JsonObject | None = diff_by_id.get(instance_id)
        if diff is None:
            raise RuntimeMatchedGateError(
                f"Gate decision has no diff row: "
                f"key={key}, instance_id={instance_id}"
            )
        expected_fields: tuple[tuple[str, JsonValue], ...] = (
            ("schema_version", GATE_DECISION_SCHEMA_VERSION),
            ("model", model),
            ("served_model", served_model),
            ("domain", domain),
            ("arm", arm),
            ("is_validation", diff["is_validation"]),
            ("decision", diff["new_decision"]),
            ("expected_skill_ids", diff["new_expected_skill_ids"]),
            ("answer_payload_hash", diff["new_answer_payload_hash"]),
            ("rerun_required", diff["rerun_required"]),
        )
        mismatches: list[str] = [
            f"{field}:expected={expected!r},observed={row.get(field)!r}"
            for field, expected in expected_fields
            if row.get(field) != expected
        ]
        if mismatches:
            raise RuntimeMatchedGateError(
                "Gate decision differs from its diff row: "
                f"instance_id={instance_id}, mismatches={mismatches}"
            )
    if seen_ids != set(diff_by_id):
        raise RuntimeMatchedGateError(
            f"Gate decision coverage mismatch: key={key}"
        )


def validate_rerun_manifest(
    manifest: Mapping[str, JsonValue],
    diff_by_id: Mapping[str, JsonObject],
    audit: Mapping[str, JsonValue],
    key: GateTaskKey,
    served_model: str,
) -> list[JsonObject]:
    """Validate the changed-row-only manifest against the audited diff."""

    model, domain, arm = key
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("schema_version", GATE_RERUN_MANIFEST_SCHEMA_VERSION),
        ("answer_schema_version", GATE_RERUN_ANSWER_SCHEMA_VERSION),
        ("model", model),
        ("served_model", served_model),
        ("domain", domain),
        ("arm", arm),
        ("old_thresholds", audit["old_thresholds"]),
        ("new_thresholds", audit["new_thresholds"]),
        ("input_artifacts", audit["input_artifacts"]),
    )
    mismatches: list[str] = [
        f"{field}:expected={expected!r},observed={manifest.get(field)!r}"
        for field, expected in expected_fields
        if manifest.get(field) != expected
    ]
    if mismatches:
        raise RuntimeMatchedGateError(
            f"Gate rerun manifest identity mismatch: key={key}, "
            f"mismatches={mismatches}"
        )
    required_answer_fields: set[str] = set(
        require_string_list(
            manifest.get("required_answer_fields"),
            f"rerun:{key}.required_answer_fields",
        )
    )
    mandatory_fields: set[str] = {
        "schema_version",
        "instance_id",
        "model",
        "served_model",
        "domain",
        "arm",
        "stage",
        "raw_output",
        "skill_ids_used",
        "expected_skill_ids",
        "failure_category",
        "answer_payload_hash",
        "execution_request_hash",
        "runtime_manifest_sha256",
        "code_bundle_sha256",
        "actual_injection_state",
        "reused_same_arm",
    }
    if not mandatory_fields.issubset(required_answer_fields):
        raise RuntimeMatchedGateError(
            "Gate rerun manifest omits required answer fields: "
            f"key={key}, missing={sorted(mandatory_fields - required_answer_fields)}"
        )
    raw_rows: list[JsonValue] = require_list(
        manifest.get("rows"),
        f"rerun:{key}.rows",
    )
    rows: list[JsonObject] = [
        require_object(row, f"rerun:{key}.rows[{index}]")
        for index, row in enumerate(raw_rows)
    ]
    rerun_count: int = require_integer(
        manifest.get("rerun_count"),
        f"rerun:{key}.rerun_count",
    )
    if rerun_count != len(rows):
        raise RuntimeMatchedGateError(
            "Gate rerun manifest count mismatch: "
            f"key={key}, declared={rerun_count}, observed={len(rows)}"
        )
    changed_ids: set[str] = {
        instance_id
        for instance_id, diff in diff_by_id.items()
        if diff["rerun_required"] is True
    }
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        context: str = f"rerun:{key}.rows[{index}]"
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        )
        if instance_id in seen_ids:
            raise RuntimeMatchedGateError(
                f"Gate rerun manifest contains duplicate instance: "
                f"key={key}, instance_id={instance_id}"
            )
        seen_ids.add(instance_id)
        diff: JsonObject | None = diff_by_id.get(instance_id)
        if diff is None or diff["rerun_required"] is not True:
            raise RuntimeMatchedGateError(
                "Gate rerun manifest includes an unchanged or unknown row: "
                f"key={key}, instance_id={instance_id}"
            )
        expected_row_fields: tuple[tuple[str, JsonValue], ...] = (
            ("model", model),
            ("served_model", served_model),
            ("domain", domain),
            ("arm", arm),
            ("new_expected_skill_ids", diff["new_expected_skill_ids"]),
            ("new_answer_payload_hash", diff["new_answer_payload_hash"]),
            ("old_request_hash", diff["old_request_hash"]),
            ("old_failure_category", diff["old_failure_category"]),
        )
        row_mismatches: list[str] = [
            f"{field}:expected={expected!r},observed={row.get(field)!r}"
            for field, expected in expected_row_fields
            if row.get(field) != expected
        ]
        if row_mismatches:
            raise RuntimeMatchedGateError(
                "Gate rerun row differs from the audited diff: "
                f"key={key}, instance_id={instance_id}, "
                f"mismatches={row_mismatches}"
            )
    if seen_ids != changed_ids:
        raise RuntimeMatchedGateError(
            "Gate rerun manifest does not exactly cover payload changes: "
            f"key={key}, missing={sorted(changed_ids - seen_ids)[:20]}, "
            f"unexpected={sorted(seen_ids - changed_ids)[:20]}"
        )
    return rows


def validate_audit_counts(
    audit: Mapping[str, JsonValue],
    diff_rows: Sequence[JsonObject],
    key: GateTaskKey,
) -> None:
    """Recompute every audit counter from per-instance evidence."""

    expected_count: int = len(diff_rows)
    decision_change_count: int = sum(
        row["decision_changed"] is True for row in diff_rows
    )
    injection_change_count: int = sum(
        row["injection_changed"] is True for row in diff_rows
    )
    payload_change_count: int = sum(
        row["payload_changed"] is True for row in diff_rows
    )
    preserved_method_failures: int = sum(
        row["rerun_required"] is False
        and row["old_failure_category"] == "method_failure"
        for row in diff_rows
    )
    expected_fields: tuple[tuple[str, JsonValue], ...] = (
        ("decision_change_count", decision_change_count),
        ("injection_change_count", injection_change_count),
        ("payload_change_count", payload_change_count),
        ("rerun_required_count", payload_change_count),
        ("preserved_row_count", expected_count - payload_change_count),
        ("preserved_method_failure_count", preserved_method_failures),
        ("old_decision_counts", _category_counts(diff_rows, "old_decision")),
        ("new_decision_counts", _category_counts(diff_rows, "new_decision")),
        ("tau1_unchanged", True),
    )
    mismatches: list[str] = [
        f"{field}:expected={expected!r},observed={audit.get(field)!r}"
        for field, expected in expected_fields
        if audit.get(field) != expected
    ]
    if mismatches:
        raise RuntimeMatchedGateError(
            f"Gate audit counter mismatch: key={key}, mismatches={mismatches}"
        )


def load_gate_task(
    audit_path: Path,
    hash_cache: dict[Path, str],
) -> GateTaskArtifacts:
    """Load and fully verify one Gate audit and all referenced artifacts."""

    resolved_audit_path: Path = audit_path.resolve()
    audit: JsonObject = load_json(resolved_audit_path, "Gate audit")
    if audit.get("schema_version") != GATE_AUDIT_SCHEMA_VERSION:
        raise RuntimeMatchedGateError(
            "Gate audit schema mismatch: "
            f"path={resolved_audit_path}, "
            f"schema={audit.get('schema_version')!r}"
        )
    if require_boolean(audit.get("valid"), "audit.valid") is not True:
        raise RuntimeMatchedGateError(
            f"Gate audit is not valid: path={resolved_audit_path}"
        )
    key, served_model = task_key_from_audit(
        audit,
        f"audit:{resolved_audit_path}",
    )
    model, domain, arm = key
    expected_count_value: int | None = RULE_DOMAIN_COUNTS.get(domain)
    if expected_count_value is None:
        raise RuntimeMatchedGateError(
            f"Gate audit uses an unknown rule domain: key={key}"
        )
    expected_count: int = require_integer(
        audit.get("expected_count"),
        "audit.expected_count",
    )
    if expected_count != expected_count_value:
        raise RuntimeMatchedGateError(
            "Gate audit denominator differs from frozen domain support: "
            f"key={key}, expected={expected_count_value}, "
            f"observed={expected_count}"
        )
    validation_count: int = require_integer(
        audit.get("validation_count"),
        "audit.validation_count",
    )
    expected_validation_count: int = max(1, int(expected_count * 0.2))
    if validation_count != expected_validation_count:
        raise RuntimeMatchedGateError(
            "Gate audit validation split size mismatch: "
            f"key={key}, expected={expected_validation_count}, "
            f"observed={validation_count}"
        )
    if require_number(audit.get("p_min"), "audit.p_min") != P_MIN:
        raise RuntimeMatchedGateError(
            f"Gate audit p_min mismatch: key={key}, "
            f"expected={P_MIN}, observed={audit.get('p_min')!r}"
        )
    old_thresholds: JsonObject = _require_thresholds(
        audit.get("old_thresholds"),
        "audit.old_thresholds",
    )
    new_thresholds: JsonObject = _require_thresholds(
        audit.get("new_thresholds"),
        "audit.new_thresholds",
    )
    if audit["old_thresholds"] != old_thresholds:
        raise RuntimeMatchedGateError(
            f"Gate audit old thresholds are not normalized: key={key}"
        )
    if audit["new_thresholds"] != new_thresholds:
        raise RuntimeMatchedGateError(
            f"Gate audit new thresholds are not normalized: key={key}"
        )
    audit_sha256: str = sha256_file(resolved_audit_path)
    hash_cache[resolved_audit_path] = audit_sha256
    verify_artifact_map(
        audit.get("input_artifacts"),
        resolved_audit_path,
        "audit.input_artifacts",
        hash_cache,
    )
    code_files: list[JsonValue] = require_list(
        audit.get("code_files"),
        "audit.code_files",
    )
    if not code_files:
        raise RuntimeMatchedGateError(
            f"Gate audit code file list is empty: key={key}"
        )
    normalized_code_files: list[JsonObject] = []
    for index, code_file_value in enumerate(code_files):
        code_file: JsonObject = require_object(
            code_file_value,
            f"audit.code_files[{index}]",
        )
        verify_artifact(
            code_file,
            resolved_audit_path,
            f"audit.code_files[{index}]",
            hash_cache,
        )
        normalized_code_files.append(code_file)
    declared_code_bundle: str = require_sha256(
        audit.get("code_bundle_sha256"),
        "audit.code_bundle_sha256",
    )
    observed_code_bundle: str = sha256_json(normalized_code_files)
    if declared_code_bundle != observed_code_bundle:
        raise RuntimeMatchedGateError(
            "Gate audit code bundle mismatch: "
            f"key={key}, expected={declared_code_bundle}, "
            f"observed={observed_code_bundle}"
        )
    output_artifacts: dict[str, tuple[Path, str]] = verify_artifact_map(
        audit.get("output_artifacts"),
        resolved_audit_path,
        "audit.output_artifacts",
        hash_cache,
    )
    required_output_names: set[str] = {"diff", "decisions", "rerun"}
    if set(output_artifacts) != required_output_names:
        raise RuntimeMatchedGateError(
            "Gate audit output inventory mismatch: "
            f"key={key}, expected={sorted(required_output_names)}, "
            f"observed={sorted(output_artifacts)}"
        )
    diff_path, diff_sha256 = output_artifacts["diff"]
    decision_path, decision_sha256 = output_artifacts["decisions"]
    rerun_path, rerun_sha256 = output_artifacts["rerun"]
    diff_rows: list[JsonObject] = load_jsonl(diff_path, "Gate diff")
    diff_by_id: dict[str, JsonObject] = validate_diff_rows(
        diff_rows,
        key,
        served_model,
        expected_count,
        old_thresholds,
        new_thresholds,
    )
    decision_rows: list[JsonObject] = load_jsonl(
        decision_path,
        "Gate decisions",
    )
    validate_decision_rows(
        decision_rows,
        diff_by_id,
        key,
        served_model,
    )
    rerun_manifest: JsonObject = load_json(
        rerun_path,
        "Gate rerun manifest",
    )
    rerun_rows: list[JsonObject] = validate_rerun_manifest(
        rerun_manifest,
        diff_by_id,
        audit,
        key,
        served_model,
    )
    validate_audit_counts(audit, diff_rows, key)
    if require_integer(
        audit.get("rerun_required_count"),
        "audit.rerun_required_count",
    ) != len(rerun_rows):
        raise RuntimeMatchedGateError(
            f"Gate audit and rerun manifest count differ: key={key}"
        )
    return {
        "key": key,
        "model": model,
        "served_model": served_model,
        "domain": domain,
        "arm": arm,
        "expected_count": expected_count,
        "audit_path": resolved_audit_path,
        "audit_sha256": audit_sha256,
        "audit": audit,
        "diff_path": diff_path,
        "diff_sha256": diff_sha256,
        "diff_rows": diff_rows,
        "decision_path": decision_path,
        "decision_sha256": decision_sha256,
        "rerun_path": rerun_path,
        "rerun_sha256": rerun_sha256,
        "rerun_rows": rerun_rows,
    }


def combine_gate_tasks(
    tasks: Sequence[GateTaskArtifacts],
) -> tuple[list[JsonObject], JsonObject]:
    """Combine validated tasks in the frozen deterministic matrix order."""

    require_expected_task_matrix([task["key"] for task in tasks])
    order: dict[GateTaskKey, int] = {
        key: index for index, key in enumerate(expected_gate_task_keys())
    }
    ordered_tasks: list[GateTaskArtifacts] = sorted(
        tasks,
        key=lambda task: order[task["key"]],
    )
    combined_diff_rows: list[JsonObject] = []
    combined_rerun_rows: list[JsonObject] = []
    task_inventory: list[JsonObject] = []
    seen_row_keys: set[GateRowKey] = set()
    for task in ordered_tasks:
        for row in task["diff_rows"]:
            row_key: GateRowKey = (
                task["model"],
                task["domain"],
                task["arm"],
                cast(str, row["instance_id"]),
            )
            if row_key in seen_row_keys:
                raise RuntimeMatchedGateError(
                    f"Combined Gate diff contains duplicate row: key={row_key}"
                )
            seen_row_keys.add(row_key)
            combined_diff_rows.append(row)
        for row in task["rerun_rows"]:
            combined_row: JsonObject = dict(row)
            combined_row["source_audit_sha256"] = task["audit_sha256"]
            combined_row["source_diff_sha256"] = task["diff_sha256"]
            combined_rerun_rows.append(combined_row)
        audit: JsonObject = task["audit"]
        task_inventory.append(
            {
                "model": task["model"],
                "served_model": task["served_model"],
                "domain": task["domain"],
                "arm": task["arm"],
                "expected_count": task["expected_count"],
                "decision_change_count": audit["decision_change_count"],
                "injection_change_count": audit["injection_change_count"],
                "payload_change_count": audit["payload_change_count"],
                "preserved_method_failure_count": (
                    audit["preserved_method_failure_count"]
                ),
                "audit": {
                    "path": str(task["audit_path"]),
                    "sha256": task["audit_sha256"],
                },
                "diff": {
                    "path": str(task["diff_path"]),
                    "sha256": task["diff_sha256"],
                },
                "decisions": {
                    "path": str(task["decision_path"]),
                    "sha256": task["decision_sha256"],
                },
                "rerun": {
                    "path": str(task["rerun_path"]),
                    "sha256": task["rerun_sha256"],
                },
            }
        )
    expected_rows: int = expected_gate_row_count()
    if len(combined_diff_rows) != expected_rows:
        raise RuntimeMatchedGateError(
            "Combined Gate row count differs from frozen support: "
            f"expected={expected_rows}, observed={len(combined_diff_rows)}"
        )
    payload_change_count: int = len(combined_rerun_rows)
    manifest: JsonObject = {
        "schema_version": GATE_COMBINED_RERUN_SCHEMA_VERSION,
        "valid": True,
        "diff_schema_version": COMBINED_DIFF_SCHEMA_VERSION,
        "answer_schema_version": GATE_RERUN_ANSWER_SCHEMA_VERSION,
        "expected_task_count": len(expected_gate_task_keys()),
        "observed_task_count": len(ordered_tasks),
        "expected_row_count": expected_rows,
        "observed_row_count": len(combined_diff_rows),
        "decision_change_count": sum(
            cast(int, task["audit"]["decision_change_count"])
            for task in ordered_tasks
        ),
        "injection_change_count": sum(
            cast(int, task["audit"]["injection_change_count"])
            for task in ordered_tasks
        ),
        "payload_change_count": payload_change_count,
        "rerun_required_count": payload_change_count,
        "preserved_row_count": expected_rows - payload_change_count,
        "preserved_method_failure_count": sum(
            cast(int, task["audit"]["preserved_method_failure_count"])
            for task in ordered_tasks
        ),
        "all_payloads_unchanged": payload_change_count == 0,
        "tasks": task_inventory,
        "source_audits_bundle_sha256": sha256_json(task_inventory),
        "rows": combined_rerun_rows,
    }
    return combined_diff_rows, manifest


def main() -> None:
    """Validate all 32 audits and emit deterministic combined evidence."""

    args = parse_args()
    audit_paths: list[Path] = [
        path.resolve() for path in cast(list[Path], args.audits)
    ]
    output_diff_path: Path = cast(Path, args.output_diff).resolve()
    output_rerun_path: Path = cast(Path, args.output_rerun).resolve()
    expected_task_count: int = len(expected_gate_task_keys())
    if len(audit_paths) != expected_task_count:
        raise RuntimeMatchedGateError(
            "Gate audit file count mismatch: "
            f"expected={expected_task_count}, observed={len(audit_paths)}"
        )
    if len(audit_paths) != len(set(audit_paths)):
        raise RuntimeMatchedGateError("Gate audit paths contain duplicates")
    if output_diff_path == output_rerun_path:
        raise RuntimeMatchedGateError(
            f"Combined Gate outputs must be distinct: path={output_diff_path}"
        )
    if output_diff_path in audit_paths or output_rerun_path in audit_paths:
        raise RuntimeMatchedGateError(
            "Combined Gate outputs must not overwrite an audit input"
        )
    hash_cache: dict[Path, str] = {}
    tasks: list[GateTaskArtifacts] = [
        load_gate_task(path, hash_cache) for path in audit_paths
    ]
    source_aliases: set[Path] = {
        path
        for path in (output_diff_path, output_rerun_path)
        if path in hash_cache
    }
    if source_aliases:
        raise RuntimeMatchedGateError(
            "Combined Gate outputs must not overwrite any verified source "
            f"artifact: aliases={sorted(source_aliases)}"
        )
    combined_diff_rows, combined_manifest = combine_gate_tasks(tasks)
    write_jsonl_atomic(output_diff_path, combined_diff_rows)
    combined_manifest["combined_diff"] = {
        "path": str(output_diff_path),
        "sha256": sha256_file(output_diff_path),
        "row_count": len(combined_diff_rows),
    }
    write_json_atomic(output_rerun_path, combined_manifest)
    print(
        canonical_json(
            {
                "event": "runtime_matched_gate_summary_complete",
                "tasks": len(tasks),
                "rows": len(combined_diff_rows),
                "payload_changes": combined_manifest[
                    "payload_change_count"
                ],
                "output_diff": str(output_diff_path),
                "output_diff_sha256": sha256_file(output_diff_path),
                "output_rerun": str(output_rerun_path),
                "output_rerun_sha256": sha256_file(output_rerun_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
