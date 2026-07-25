#!/usr/bin/env python3
"""Import verified provisional K=2 answers into the audited record schema."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    FailureCategory,
    JsonLike,
    JsonObject,
    JsonValue,
    RuntimeManifest,
    SemanticArm,
    audit_record_coverage,
    canonical_json,
    classify_request_error,
    normalize_skill_ids,
    sha256_file,
    sha256_text,
)
from scripts.audit_k2_reuse import (
    AnswerRuntime,
    answer_hash,
    expected_skill_ids,
    load_answer_runtime,
    load_corpus,
    load_decisions,
    load_instances,
    load_manifest,
    loaded_skills,
    require_string,
)
from scripts.run_k2_answers import (
    AnswerLine,
    AuditRow,
    failure_record,
    indexed_answer_lines,
    load_audit,
    load_jsonl,
    success_record,
)


STATUS_CODE_PATTERN: re.Pattern[str] = re.compile(
    r"(?:error code|status code)\s*:\s*(\d{3})",
    re.IGNORECASE,
)


class ImportedEngineResult(NamedTuple):
    """Fields required by the shared successful-answer record builder."""

    raw_output: str
    transcript: str | None
    skill_ids_used: list[str]
    meta: JsonObject


def parse_args() -> argparse.Namespace:
    """Parse one explicit provisional-answer import job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--decision-source", required=True, type=Path)
    parser.add_argument("--provisional", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
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
    return parser.parse_args()


def status_code_from_error(message: str) -> int | None:
    """Extract one explicit HTTP status code from a legacy error string."""

    match: re.Match[str] | None = STATUS_CODE_PATTERN.search(message)
    return int(match.group(1)) if match is not None else None


def optional_string(value: JsonValue | None, context: str) -> str | None:
    """Return one optional string or raise on a different type."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise DownstreamDataError(
            f"Expected optional string: context={context}, "
            f"actual={type(value).__name__}"
        )
    return value


def source_provenance(
    source_path: Path,
    source_sha256: str,
    line_number: int,
    raw_line: str,
) -> JsonObject:
    """Build immutable provenance for one provisional source row."""

    return {
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_line_number": line_number,
        "source_line_sha256": sha256_text(raw_line.rstrip("\r\n")),
    }


def imported_success_record(
    provisional: Mapping[str, JsonLike],
    instance: JsonObject,
    arm: SemanticArm,
    model: str,
    model_label: str,
    expected: Sequence[str],
    request_hash: str,
    manifest: RuntimeManifest,
    decision_source_sha256: str,
) -> JsonObject:
    """Upgrade one non-empty provisional response after identity checks."""

    actual_skill_ids: tuple[str, ...] = normalize_skill_ids(
        provisional.get("skill_ids_used"),
        "provisional.skill_ids_used",
    )
    if actual_skill_ids != tuple(expected):
        raise DownstreamDataError(
            "Provisional answer injection mismatch: "
            f"instance_id={instance.get('instance_id')!r}, "
            f"expected={tuple(expected)}, actual={actual_skill_ids}"
        )
    raw_output: str = require_string(
        provisional.get("raw_output"),
        "provisional.raw_output",
    )
    if not raw_output.strip():
        raise DownstreamDataError(
            "Provisional successful answer is empty: "
            f"instance_id={instance.get('instance_id')!r}"
        )
    transcript: str | None = optional_string(
        provisional.get("transcript"),
        "provisional.transcript",
    )
    result = ImportedEngineResult(
        raw_output=raw_output,
        transcript=transcript,
        skill_ids_used=list(actual_skill_ids),
        meta={},
    )
    return success_record(
        instance,
        arm,
        model,
        model_label,
        expected,
        request_hash,
        result,
        1,
        manifest,
        decision_source_sha256,
    )


def imported_method_failure_record(
    instance: JsonObject,
    arm: SemanticArm,
    model: str,
    model_label: str,
    expected: Sequence[str],
    request_hash: str,
    manifest: RuntimeManifest,
    decision_source_sha256: str,
    error_message: str,
    status_code: int | None,
) -> JsonObject:
    """Upgrade one deterministic provisional request failure."""

    return failure_record(
        instance,
        arm,
        model,
        model_label,
        expected,
        request_hash,
        "method_failure",
        {
            "exception_name": "ProvisionalAnswerError",
            "message": error_message,
            "status_code": status_code,
            "response_body": error_message,
        },
        1,
        manifest,
        decision_source_sha256,
    )


def write_jsonl_atomic(
    path: Path,
    records: Sequence[Mapping[str, JsonLike]],
) -> None:
    """Atomically write canonical JSONL records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(canonical_json(record) + "\n")
    temporary_path.replace(path)


def main() -> None:
    """Import successes and method failures while leaving retries missing."""

    args = parse_args()
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    decision_source_path: Path = cast(Path, args.decision_source).resolve()
    provisional_path: Path = cast(Path, args.provisional).resolve()
    audit_path: Path = cast(Path, args.audit).resolve()
    manifest_path: Path = cast(Path, args.runtime_manifest).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    model: str = str(args.model)
    model_label: str = str(args.model_label)
    arm: SemanticArm = cast(SemanticArm, str(args.arm))
    domain: str = str(args.domain)
    expected_count: int = int(args.expected_count)

    runtime: AnswerRuntime = load_answer_runtime()
    instances: list[JsonObject] = load_instances(instances_path, domain)
    if len(instances) != expected_count:
        raise DownstreamDataError(
            "Provisional import denominator mismatch: "
            f"expected={expected_count}, actual={len(instances)}"
        )
    instance_index: dict[str, JsonObject] = {
        require_string(instance.get("instance_id"), "instance.instance_id"): instance
        for instance in instances
    }
    corpus: dict[str, JsonObject] = load_corpus(corpus_path)
    decisions: dict[str, JsonObject] = load_decisions(decision_source_path)
    audit: dict[str, AuditRow] = load_audit(audit_path, arm)
    provisional_lines: list[AnswerLine] = load_jsonl(
        provisional_path,
        "provisional-answer",
    )
    provisional_index: dict[str, AnswerLine] = indexed_answer_lines(
        provisional_lines,
        "provisional-answer",
    )
    for name, observed_ids in (
        ("decision-source", list(decisions)),
        ("reuse-audit", list(audit)),
        ("provisional-answer", list(provisional_index)),
    ):
        coverage = audit_record_coverage(list(instance_index), observed_ids)
        if not coverage.complete:
            raise DownstreamDataError(
                f"{name} coverage mismatch: "
                f"missing={list(coverage.missing_ids)[:20]}, "
                f"duplicates={list(coverage.duplicate_ids)[:20]}, "
                f"unexpected={list(coverage.unexpected_ids)[:20]}"
            )
    manifest: RuntimeManifest = load_manifest(
        manifest_path,
        sha256_file(instances_path),
        sha256_file(corpus_path),
    )
    if manifest["runtime_identity"].get("served_model") != model:
        raise DownstreamDataError(
            "Import model does not match runtime manifest: "
            f"cli_model={model}, "
            f"manifest_served_model={manifest['runtime_identity'].get('served_model')!r}"
        )

    source_sha256: str = sha256_file(provisional_path)
    decision_source_sha256: str = sha256_file(decision_source_path)
    imported_records: list[JsonObject] = []
    category_counts: Counter[str] = Counter()
    for line_number, provisional_line in enumerate(
        provisional_lines,
        start=1,
    ):
        provisional: JsonObject = provisional_line["record"]
        instance_id: str = require_string(
            provisional.get("instance_id"),
            "provisional.instance_id",
        )
        if provisional.get("dataset") != domain:
            raise DownstreamDataError(
                "Provisional dataset mismatch: "
                f"instance_id={instance_id}, expected={domain}, "
                f"actual={provisional.get('dataset')!r}"
            )
        if provisional.get("method") != arm:
            raise DownstreamDataError(
                "Provisional arm mismatch: "
                f"instance_id={instance_id}, expected={arm}, "
                f"actual={provisional.get('method')!r}"
            )
        instance: JsonObject = instance_index[instance_id]
        expected: tuple[str, ...] = expected_skill_ids(
            arm,
            decisions[instance_id],
        )
        if list(expected) != audit[instance_id]["expected_skill_ids"]:
            raise DownstreamDataError(
                "Provisional import audit is stale: "
                f"instance_id={instance_id}"
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
        if request_hash != audit[instance_id]["new_request_hash"]:
            raise DownstreamDataError(
                "Provisional import request hash mismatch: "
                f"instance_id={instance_id}"
            )
        raw_error: JsonValue | None = provisional.get("error")
        raw_output: JsonValue | None = provisional.get("raw_output")
        imported: JsonObject | None = None
        category: str
        if isinstance(raw_error, str) and raw_error:
            status_code: int | None = status_code_from_error(raw_error)
            failure_category: FailureCategory = classify_request_error(
                "ProvisionalAnswerError",
                raw_error,
                status_code,
                raw_error,
            )
            if failure_category == "method_failure":
                imported = imported_method_failure_record(
                    instance,
                    arm,
                    model,
                    model_label,
                    expected,
                    request_hash,
                    manifest,
                    decision_source_sha256,
                    raw_error,
                    status_code,
                )
                category = "method_failure"
            else:
                category = "needs_rerun_error"
        elif isinstance(raw_output, str) and raw_output.strip():
            imported = imported_success_record(
                provisional,
                instance,
                arm,
                model,
                model_label,
                expected,
                request_hash,
                manifest,
                decision_source_sha256,
            )
            category = "success"
        else:
            category = "needs_rerun_empty"
        category_counts[category] += 1
        if imported is not None:
            imported["provisional_source"] = source_provenance(
                provisional_path,
                source_sha256,
                line_number,
                provisional_line["raw_line"],
            )
            imported_records.append(imported)

    write_jsonl_atomic(output_path, imported_records)
    summary: JsonObject = {
        "event": "k2_provisional_answers_imported",
        "model": model,
        "domain": domain,
        "arm": arm,
        "expected": expected_count,
        "imported": len(imported_records),
        "pending_rerun": expected_count - len(imported_records),
        "category_counts": dict(sorted(category_counts.items())),
        "provisional_source": str(provisional_path),
        "provisional_source_sha256": source_sha256,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
