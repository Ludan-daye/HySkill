#!/usr/bin/env python3
"""Export path-free public packs for fresh runtime-matched baselines."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

from hyskill.runtime_matched_execution import canonical_json, sha256_file
from hyskill.runtime_matched_public import (
    JsonObject,
    JsonValue,
    deterministic_jsonl_gzip,
    public_usage_job,
)
from scripts.validate_runtime_matched_baselines import (
    COMPARISONS_SCHEMA_VERSION,
    DOMAINS,
    EXPECTED_ANSWER_ROWS,
    EXPECTED_DECISION_ROWS,
    EXPECTED_LONG_METRICS,
    EXPECTED_USAGE_JOBS,
    SUMMARY_SCHEMA_VERSION,
    USAGE_SUMMARY_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
    parse_model_list,
    require_list,
    require_object,
    require_string,
    validate_bare_model,
    validate_native_model,
)


JsonScalar: TypeAlias = None | bool | int | float | str

MODEL_PACK_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-public-model-pack-v1"
)
FLEET_PACK_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-public-fleet-pack-v1"
)
MODEL_METRICS_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-public-model-metrics-v1"
)
PUBLIC_USAGE_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-public-usage-summary-v1"
)
PUBLIC_VALIDATION_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-public-validation-summary-v1"
)
PUBLIC_MODEL_VALIDATION_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-public-model-validation-v1"
)
DOMAIN_ORDER: dict[str, int] = {
    domain: index for index, (domain, _count, _heldout) in enumerate(DOMAINS)
}
ARM_ORDER: dict[str, int] = {
    "bare": 0,
    "always_rerank": 1,
    "select_bm25": 2,
}
STAGE_ORDER: dict[str, int] = {"decision": 0, "answer": 1}
SENSITIVE_TOKENS: tuple[bytes, ...] = (
    b"/root/",
    b"/home/",
    b"/Users/",
    b"127.0.0.1",
    b"localhost",
    b'"api_base"',
    b'"api_key"',
    b'"checkpoint_path"',
    b'"evaluator"',
    b'"gpu_uuid"',
    b'"ground_truth"',
    b'"password"',
    b'"process_command"',
    b'"raw_output"',
    b'"raw_response"',
    b'"raw_responses"',
    b"GPU-",
    b"BEGIN OPENSSH",
)


class RuntimeMatchedExportError(ValueError):
    """Raised when validated private evidence cannot form a public pack."""


def parse_args() -> argparse.Namespace:
    """Parse the complete validated export inventory."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--seven-models", required=True)
    parser.add_argument("--five-models", required=True)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--usage-summary", required=True, type=Path)
    parser.add_argument("--metrics-long", required=True, type=Path)
    parser.add_argument("--metrics-summary", required=True, type=Path)
    parser.add_argument("--comparisons", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path, context: str) -> JsonObject:
    """Load one required JSON object."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: path={path}")
    try:
        value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise RuntimeMatchedExportError(
            f"{context} is malformed: path={path}, line={error.lineno}, "
            f"column={error.colno}, message={error.msg}"
        ) from error
    return require_object(value, context)


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load one regular or gzip JSONL file."""

    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: path={path}")
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[JsonObject] = []
    with opener(path, "rt", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise RuntimeMatchedExportError(
                    f"{context} contains a blank line: "
                    f"path={path}, line={line_number}"
                )
            try:
                value: JsonValue = cast(JsonValue, json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeMatchedExportError(
                    f"{context} is malformed: path={path}, "
                    f"line={line_number}, column={error.colno}, "
                    f"message={error.msg}"
                ) from error
            rows.append(
                require_object(value, f"{context}:{path}:{line_number}")
            )
    if not rows:
        raise RuntimeMatchedExportError(
            f"{context} contains no rows: path={path}"
        )
    return rows


def strip_paths(value: JsonValue) -> JsonValue:
    """Recursively remove private path-valued fields."""

    if isinstance(value, list):
        return [strip_paths(item) for item in value]
    if isinstance(value, dict):
        return {
            key: strip_paths(item)
            for key, item in value.items()
            if key not in {"path", "marker_path"}
        }
    return value


def require_sha_binding(
    record: Mapping[str, JsonValue],
    path: Path,
    context: str,
) -> None:
    """Require one validation-record SHA to match the supplied file."""

    expected: str = require_string(record.get("sha256"), f"{context}.sha256")
    actual: str = sha256_file(path)
    if expected != actual:
        raise RuntimeMatchedExportError(
            f"Validated input SHA mismatch: context={context}, "
            f"expected={expected}, actual={actual}, path={path}"
        )


def model_report_index(
    validation: Mapping[str, JsonValue],
    seven_models: Sequence[str],
) -> dict[str, JsonObject]:
    """Index the seven validated model reports."""

    reports: dict[str, JsonObject] = {}
    for index, item in enumerate(
        require_list(validation.get("models"), "validation.models")
    ):
        report: JsonObject = require_object(
            item,
            f"validation.models[{index}]",
        )
        model: str = require_string(
            report.get("model"),
            f"validation.models[{index}].model",
        )
        if model in reports:
            raise RuntimeMatchedExportError(
                f"Validation contains duplicate model: model={model}"
            )
        reports[model] = report
    if set(reports) != set(seven_models):
        raise RuntimeMatchedExportError(
            f"Validation model support mismatch: "
            f"missing={sorted(set(seven_models) - set(reports))}, "
            f"unexpected={sorted(set(reports) - set(seven_models))}"
        )
    return reports


def validate_export_inputs(
    validation_path: Path,
    usage_path: Path,
    metrics_long_path: Path,
    metrics_summary_path: Path,
    comparisons_path: Path,
    result_root: Path,
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> tuple[JsonObject, dict[str, JsonObject]]:
    """Bind all export inputs to the completed fleet validation report."""

    validation: JsonObject = load_json(
        validation_path,
        "fleet validation report",
    )
    if (
        validation.get("schema_version") != VALIDATION_SCHEMA_VERSION
        or validation.get("valid") is not True
    ):
        raise RuntimeMatchedExportError(
            f"Fleet validation report is not valid: path={validation_path}"
        )
    support: JsonObject = require_object(
        validation.get("support"),
        "validation.support",
    )
    if (
        support.get("seven_models") != list(seven_models)
        or support.get("five_models") != list(five_models)
    ):
        raise RuntimeMatchedExportError(
            f"Fleet validation support differs from export support: "
            f"validation={support}"
        )
    totals: JsonObject = require_object(
        validation.get("totals"),
        "validation.totals",
    )
    if (
        totals.get("answer_rows") != EXPECTED_ANSWER_ROWS
        or totals.get("decision_rows") != EXPECTED_DECISION_ROWS
        or totals.get("usage_jobs") != EXPECTED_USAGE_JOBS
    ):
        raise RuntimeMatchedExportError(
            f"Fleet validation totals are incomplete: totals={totals}"
        )
    usage_record: JsonObject = require_object(
        validation.get("usage"),
        "validation.usage",
    )
    require_sha_binding(usage_record, usage_path, "validation.usage")
    aggregates: JsonObject = require_object(
        validation.get("aggregates"),
        "validation.aggregates",
    )
    for name, path in (
        ("metrics_long", metrics_long_path),
        ("metrics_summary", metrics_summary_path),
        ("comparisons", comparisons_path),
    ):
        require_sha_binding(
            require_object(
                aggregates.get(name),
                f"validation.aggregates.{name}",
            ),
            path,
            f"validation.aggregates.{name}",
        )
    reports: dict[str, JsonObject] = model_report_index(
        validation,
        seven_models,
    )
    for model, report in reports.items():
        marker_path: Path = result_root / model / "FORMAL_COMPLETE"
        if not marker_path.is_file():
            raise FileNotFoundError(
                f"Formal completion marker is missing: model={model}, "
                f"path={marker_path}"
            )
        expected_marker_sha: str = require_string(
            report.get("marker_sha256"),
            f"validation.models.{model}.marker_sha256",
        )
        actual_marker_sha: str = sha256_file(marker_path)
        if expected_marker_sha != actual_marker_sha:
            raise RuntimeMatchedExportError(
                f"Formal marker SHA mismatch: model={model}, "
                f"expected={expected_marker_sha}, actual={actual_marker_sha}"
            )
    return validation, reports


def answer_sort_key(row: Mapping[str, JsonValue]) -> tuple[int, int, str]:
    """Return deterministic answer ordering."""

    domain: str = require_string(row.get("domain"), "answer.domain")
    arm: str = require_string(row.get("arm"), "answer.arm")
    return (
        DOMAIN_ORDER[domain],
        ARM_ORDER[arm],
        require_string(row.get("instance_id"), "answer.instance_id"),
    )


def decision_sort_key(row: Mapping[str, JsonValue]) -> tuple[int, int, str]:
    """Return deterministic decision ordering."""

    return answer_sort_key(row)


def runtime_sort_key(
    row: Mapping[str, JsonValue],
) -> tuple[int, int, int, str]:
    """Return deterministic runtime-job ordering."""

    job: JsonObject = require_object(row.get("job"), "runtime.job")
    domain: str = require_string(job.get("domain"), "runtime.job.domain")
    arm: str = require_string(job.get("arm"), "runtime.job.arm")
    job_id: str = require_string(job.get("job_id"), "runtime.job.job_id")
    stage: str = (
        "decision"
        if "-decision-" in job_id or job_id.endswith("-decision")
        else "answer"
    )
    return (
        DOMAIN_ORDER[domain],
        ARM_ORDER[arm],
        STAGE_ORDER[stage],
        job_id,
    )


def usage_sort_key(
    row: Mapping[str, JsonValue],
) -> tuple[int, int, int]:
    """Return deterministic usage-job ordering."""

    domain: str = require_string(row.get("domain"), "usage.domain")
    arm: str = require_string(row.get("arm"), "usage.arm")
    stage: str = require_string(row.get("stage"), "usage.stage")
    return (
        DOMAIN_ORDER[domain],
        ARM_ORDER[arm],
        STAGE_ORDER[stage],
    )


def metric_sort_key(
    row: Mapping[str, JsonValue],
) -> tuple[str, str, int, str]:
    """Return deterministic long-metric ordering."""

    raw_domain: JsonValue | None = row.get("domain")
    domain_index: int = (
        len(DOMAIN_ORDER)
        if raw_domain is None
        else DOMAIN_ORDER[require_string(raw_domain, "metric.domain")]
    )
    return (
        require_string(row.get("split"), "metric.split"),
        require_string(row.get("level"), "metric.level"),
        domain_index,
        require_string(row.get("arm"), "metric.arm"),
    )


def write_json(path: Path, payload: JsonValue) -> None:
    """Write one canonical JSON value."""

    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def write_jsonl_gzip(
    path: Path,
    rows: Sequence[Mapping[str, JsonValue]],
) -> None:
    """Write reproducible canonical JSONL gzip bytes."""

    path.write_bytes(deterministic_jsonl_gzip(rows))


def output_record(
    directory: Path,
    filename: str,
    rows: int,
    schema: str,
    provenance: str,
) -> JsonObject:
    """Return one generated-file manifest record."""

    path: Path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Generated public artifact is missing: path={path}"
        )
    return {
        "sha256": sha256_file(path),
        "rows": rows,
        "schema": schema,
        "provenance_level": provenance,
    }


def scan_sensitive_file(path: Path) -> None:
    """Reject server paths, endpoint details, and credential markers."""

    raw_bytes: bytes = path.read_bytes()
    inspected: bytes = (
        gzip.decompress(raw_bytes) if path.suffix == ".gz" else raw_bytes
    )
    matches: list[str] = [
        token.decode("utf-8", errors="replace")
        for token in SENSITIVE_TOKENS
        if token in inspected
    ]
    if matches:
        raise RuntimeMatchedExportError(
            f"Generated public artifact contains sensitive tokens: "
            f"path={path}, matches={matches}"
        )


def scan_pack(directory: Path) -> None:
    """Scan every generated public-pack file."""

    for path in sorted(directory.iterdir()):
        if path.is_file():
            scan_sensitive_file(path)


def model_readme(
    model: str,
    native_eligible: bool,
    answer_rows: int,
    decision_rows: int,
) -> str:
    """Return one concise model-pack README."""

    native_line: str = (
        "- Native Rerank and BM25 Select are available on all four domains."
        if native_eligible
        else (
            "- Native Rerank and BM25 Select are unavailable because this "
            "model cannot support the frozen 50-candidate prompt; cells are "
            "not zero-filled."
        )
    )
    return "\n".join(
        (
            f"# Runtime-matched baseline pack: {model}",
            "",
            "This pack contains fresh baseline evidence generated under the "
            "same checkpoint, tokenizer, chat template, vLLM, BF16, and "
            "8,192-token runtime contract as the active K=2 experiment.",
            "",
            "## Coverage",
            "",
            f"- Public answer rows: {answer_rows}.",
            f"- Public decision rows: {decision_rows}.",
            "- Bare is available on all four rule-scored domains.",
            native_line,
            "",
            "## Evidence policy",
            "",
            "- No legacy compact baseline row is included.",
            "- Actual service-reported usage is retained without imputation.",
            "- Deterministic method failures remain in the denominator.",
            "- Raw model text, gold answers, endpoints, server paths, GPU "
            "identifiers, and credentials are omitted.",
            "",
        )
    )


def arm_availability(native_eligible: bool) -> JsonObject:
    """Describe exact per-model baseline support."""

    native: JsonObject = (
        {
            "status": "available",
            "domains": list(DOMAIN_ORDER),
        }
        if native_eligible
        else {
            "status": "unavailable",
            "domains": [],
            "reason": (
                "The frozen 50-candidate Rerank/Select prompt is unsupported."
            ),
            "accounting": "excluded from native-arm denominators; never zero-filled",
        }
    )
    return {
        "bare": {
            "status": "available",
            "domains": list(DOMAIN_ORDER),
        },
        "always_rerank": native,
        "select_bm25": dict(native),
    }


def model_metrics(
    model: str,
    long_rows: Sequence[JsonObject],
    summary: Mapping[str, JsonValue],
) -> JsonObject:
    """Build one path-free model metric product."""

    selected: list[JsonObject] = [
        row for row in long_rows if row.get("model") == model
    ]
    model_pooled: JsonObject = require_object(
        require_object(
            summary.get("model_pooled"),
            "metrics-summary.model_pooled",
        ).get(model),
        f"metrics-summary.model_pooled.{model}",
    )
    return {
        "schema_version": MODEL_METRICS_SCHEMA_VERSION,
        "model": model,
        "model_pooled": model_pooled,
        "metric_rows": sorted(selected, key=metric_sort_key),
    }


def build_model_pack(
    directory: Path,
    result_root: Path,
    model: str,
    native_eligible: bool,
    model_report: Mapping[str, JsonValue],
    usage_jobs: Sequence[JsonObject],
    long_rows: Sequence[JsonObject],
    metrics_summary: Mapping[str, JsonValue],
    validation_sha: str,
) -> None:
    """Build and validate one complete model public pack."""

    directory.mkdir(parents=True, exist_ok=False)
    model_root: Path = result_root / model
    answer_rows, runtime_rows, _bare_sources = validate_bare_model(
        model_root,
        model,
    )
    decision_rows: list[JsonObject] = []
    if native_eligible:
        (
            native_answers,
            native_decisions,
            native_runtime,
            _native_sources,
        ) = validate_native_model(model_root, model)
        answer_rows.extend(native_answers)
        decision_rows.extend(native_decisions)
        runtime_rows.extend(native_runtime)
    ordered_answers: list[JsonObject] = sorted(
        answer_rows,
        key=answer_sort_key,
    )
    ordered_decisions: list[JsonObject] = sorted(
        decision_rows,
        key=decision_sort_key,
    )
    ordered_runtime: list[JsonObject] = sorted(
        runtime_rows,
        key=runtime_sort_key,
    )
    ordered_usage: list[JsonObject] = sorted(
        [public_usage_job(row) for row in usage_jobs],
        key=usage_sort_key,
    )
    expected_answers: int = 8490 if native_eligible else 2830
    expected_decisions: int = 5660 if native_eligible else 0
    expected_jobs: int = 20 if native_eligible else 4
    if (
        len(ordered_answers) != expected_answers
        or len(ordered_decisions) != expected_decisions
        or len(ordered_runtime) != expected_jobs
        or len(ordered_usage) != expected_jobs
    ):
        raise RuntimeMatchedExportError(
            f"Model public count mismatch: model={model}, "
            f"answers={len(ordered_answers)}/{expected_answers}, "
            f"decisions={len(ordered_decisions)}/{expected_decisions}, "
            f"runtime={len(ordered_runtime)}/{expected_jobs}, "
            f"usage={len(ordered_usage)}/{expected_jobs}"
        )
    write_jsonl_gzip(
        directory / "answer_per_instance.jsonl.gz",
        ordered_answers,
    )
    if native_eligible:
        write_jsonl_gzip(
            directory / "decision_per_instance.jsonl.gz",
            ordered_decisions,
        )
    write_jsonl_gzip(directory / "runtime_jobs.jsonl.gz", ordered_runtime)
    write_jsonl_gzip(directory / "usage_jobs.jsonl.gz", ordered_usage)
    metrics: JsonObject = model_metrics(
        model,
        long_rows,
        metrics_summary,
    )
    write_json(directory / "metrics.json", metrics)
    public_validation: JsonObject = require_object(
        strip_paths(
            {
                "schema_version": PUBLIC_MODEL_VALIDATION_SCHEMA_VERSION,
                **dict(model_report),
                "fresh_only": True,
                "legacy_compact_baseline_read": False,
            }
        ),
        f"public-model-validation:{model}",
    )
    write_json(directory / "validation.json", public_validation)
    readme: str = model_readme(
        model,
        native_eligible,
        len(ordered_answers),
        len(ordered_decisions),
    )
    (directory / "README.md").write_text(readme, encoding="utf-8")
    files: JsonObject = {
        "README.md": output_record(
            directory,
            "README.md",
            len(readme.splitlines()),
            "markdown-v1",
            "documentation",
        ),
        "answer_per_instance.jsonl.gz": output_record(
            directory,
            "answer_per_instance.jsonl.gz",
            len(ordered_answers),
            "runtime-matched-public-answer-row-v1",
            "per_instance",
        ),
        "metrics.json": output_record(
            directory,
            "metrics.json",
            len(cast(list[JsonValue], metrics["metric_rows"])),
            MODEL_METRICS_SCHEMA_VERSION,
            "per_model",
        ),
        "runtime_jobs.jsonl.gz": output_record(
            directory,
            "runtime_jobs.jsonl.gz",
            len(ordered_runtime),
            "runtime-matched-public-runtime-job-v1",
            "per_job",
        ),
        "usage_jobs.jsonl.gz": output_record(
            directory,
            "usage_jobs.jsonl.gz",
            len(ordered_usage),
            "runtime-matched-public-usage-job-v1",
            "per_job",
        ),
        "validation.json": output_record(
            directory,
            "validation.json",
            1,
            PUBLIC_MODEL_VALIDATION_SCHEMA_VERSION,
            "audit",
        ),
    }
    if native_eligible:
        files["decision_per_instance.jsonl.gz"] = output_record(
            directory,
            "decision_per_instance.jsonl.gz",
            len(ordered_decisions),
            "runtime-matched-public-decision-row-v1",
            "per_instance",
        )
    manifest: JsonObject = {
        "schema_version": MODEL_PACK_SCHEMA_VERSION,
        "model": model,
        "domains": list(DOMAIN_ORDER),
        "arms": arm_availability(native_eligible),
        "fresh_only": True,
        "legacy_compact_baseline_read": False,
        "files": {
            name: files[name] for name in sorted(files)
        },
        "sources": {
            "fleet_validation_sha256": validation_sha,
            "formal_complete_sha256": require_string(
                model_report.get("marker_sha256"),
                f"model-report:{model}.marker_sha256",
            ),
            "runtime_identity_sha256": require_string(
                model_report.get("runtime_identity_sha256"),
                f"model-report:{model}.runtime_identity_sha256",
            ),
        },
        "manifest_self_policy": {
            "included_in_files": False,
            "reason": (
                "manifest.json is excluded because a file cannot contain "
                "its own stable SHA-256 digest."
            ),
        },
    }
    write_json(directory / "manifest.json", manifest)
    scan_pack(directory)


def public_usage_summary(
    private: Mapping[str, JsonValue],
) -> JsonObject:
    """Build a path-free fleet usage product."""

    if private.get("schema_version") != USAGE_SUMMARY_SCHEMA_VERSION:
        raise RuntimeMatchedExportError(
            f"Unexpected usage schema: value={private.get('schema_version')!r}"
        )
    jobs: list[JsonObject] = [
        public_usage_job(
            require_object(item, f"usage.jobs[{index}]")
        )
        for index, item in enumerate(
            require_list(private.get("jobs"), "usage.jobs")
        )
    ]
    if len(jobs) != EXPECTED_USAGE_JOBS:
        raise RuntimeMatchedExportError(
            f"Usage job count mismatch: expected={EXPECTED_USAGE_JOBS}, "
            f"actual={len(jobs)}"
        )
    return {
        "schema_version": PUBLIC_USAGE_SCHEMA_VERSION,
        "support": cast(JsonValue, private.get("support")),
        "overall": cast(JsonValue, private.get("overall")),
        "by_model_arm_stage": cast(
            JsonValue,
            private.get("by_model_arm_stage"),
        ),
        "by_arm_stage": cast(JsonValue, private.get("by_arm_stage")),
        "jobs": jobs,
        "semantics": cast(JsonValue, private.get("semantics")),
    }


def fleet_readme() -> str:
    """Return the fleet public-pack README."""

    return "\n".join(
        (
            "# Runtime-matched baseline fleet",
            "",
            "This directory contains fresh Bare, native Rerank, and BM25 "
            "Select evidence matched to the active K=2 runtime identities.",
            "",
            "The five data products are:",
            "",
            "- `metrics_long.jsonl.gz`",
            "- `metrics_summary.json`",
            "- `paired_comparisons.json`",
            "- `usage_summary.json`",
            "- `validation_summary.json`",
            "",
            "DeepSeek-7B and Yi-1.5-9B native 50-candidate arms are unavailable "
            "and are never represented as zero. Method failures remain "
            "incorrect. Token counts come only from actual service responses; "
            "missing usage is never imputed.",
            "",
        )
    )


def build_fleet_pack(
    directory: Path,
    validation: Mapping[str, JsonValue],
    usage: Mapping[str, JsonValue],
    long_rows: Sequence[JsonObject],
    metrics_summary: Mapping[str, JsonValue],
    comparisons: Mapping[str, JsonValue],
    model_manifest_shas: Mapping[str, str],
    validation_sha: str,
) -> None:
    """Build and validate the five-product fleet public pack."""

    directory.mkdir(parents=True, exist_ok=False)
    if len(long_rows) != EXPECTED_LONG_METRICS:
        raise RuntimeMatchedExportError(
            f"Metric row count mismatch: expected={EXPECTED_LONG_METRICS}, "
            f"actual={len(long_rows)}"
        )
    if metrics_summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise RuntimeMatchedExportError(
            "Unexpected metrics summary schema: "
            f"value={metrics_summary.get('schema_version')!r}"
        )
    if comparisons.get("schema_version") != COMPARISONS_SCHEMA_VERSION:
        raise RuntimeMatchedExportError(
            "Unexpected comparisons schema: "
            f"value={comparisons.get('schema_version')!r}"
        )
    write_jsonl_gzip(
        directory / "metrics_long.jsonl.gz",
        sorted(long_rows, key=metric_sort_key),
    )
    write_json(
        directory / "metrics_summary.json",
        strip_paths(cast(JsonValue, dict(metrics_summary))),
    )
    write_json(
        directory / "paired_comparisons.json",
        strip_paths(cast(JsonValue, dict(comparisons))),
    )
    public_usage: JsonObject = public_usage_summary(usage)
    write_json(directory / "usage_summary.json", public_usage)
    public_validation: JsonObject = require_object(
        strip_paths(
            {
                "schema_version": PUBLIC_VALIDATION_SCHEMA_VERSION,
                "support": cast(JsonValue, validation.get("support")),
                "totals": cast(JsonValue, validation.get("totals")),
                "models": cast(JsonValue, validation.get("models")),
                "usage": cast(JsonValue, validation.get("usage")),
                "aggregates": cast(JsonValue, validation.get("aggregates")),
                "semantics": cast(JsonValue, validation.get("semantics")),
                "valid": cast(JsonValue, validation.get("valid")),
            }
        ),
        "public-validation-summary",
    )
    write_json(directory / "validation_summary.json", public_validation)
    readme: str = fleet_readme()
    (directory / "README.md").write_text(readme, encoding="utf-8")
    files: JsonObject = {
        "README.md": output_record(
            directory,
            "README.md",
            len(readme.splitlines()),
            "markdown-v1",
            "documentation",
        ),
        "metrics_long.jsonl.gz": output_record(
            directory,
            "metrics_long.jsonl.gz",
            len(long_rows),
            "runtime-matched-baseline-metric-v1",
            "fleet",
        ),
        "metrics_summary.json": output_record(
            directory,
            "metrics_summary.json",
            1,
            SUMMARY_SCHEMA_VERSION,
            "fleet",
        ),
        "paired_comparisons.json": output_record(
            directory,
            "paired_comparisons.json",
            4,
            COMPARISONS_SCHEMA_VERSION,
            "fleet",
        ),
        "usage_summary.json": output_record(
            directory,
            "usage_summary.json",
            EXPECTED_USAGE_JOBS,
            PUBLIC_USAGE_SCHEMA_VERSION,
            "fleet",
        ),
        "validation_summary.json": output_record(
            directory,
            "validation_summary.json",
            1,
            PUBLIC_VALIDATION_SCHEMA_VERSION,
            "audit",
        ),
    }
    manifest: JsonObject = {
        "schema_version": FLEET_PACK_SCHEMA_VERSION,
        "support": cast(JsonValue, validation.get("support")),
        "totals": cast(JsonValue, validation.get("totals")),
        "five_data_products": [
            "metrics_long.jsonl.gz",
            "metrics_summary.json",
            "paired_comparisons.json",
            "usage_summary.json",
            "validation_summary.json",
        ],
        "model_packs": {
            model: {"manifest_sha256": model_manifest_shas[model]}
            for model in sorted(model_manifest_shas)
        },
        "files": {name: files[name] for name in sorted(files)},
        "sources": {"fleet_validation_sha256": validation_sha},
        "manifest_self_policy": {
            "included_in_files": False,
            "reason": (
                "manifest.json is excluded because a file cannot contain "
                "its own stable SHA-256 digest."
            ),
        },
    }
    write_json(directory / "manifest.json", manifest)
    scan_pack(directory)


def ensure_destinations_absent(
    output_root: Path,
    seven_models: Sequence[str],
) -> dict[str, Path]:
    """Resolve all immutable output destinations before creating anything."""

    if not output_root.is_dir():
        raise FileNotFoundError(
            f"Output root directory does not exist: path={output_root}"
        )
    destinations: dict[str, Path] = {
        model: output_root / model / "baselines-runtime-matched"
        for model in seven_models
    }
    destinations["fleet"] = (
        output_root / "baselines-runtime-matched-fleet"
    )
    missing_parents: list[str] = [
        str(path.parent)
        for path in destinations.values()
        if not path.parent.is_dir()
    ]
    if missing_parents:
        raise FileNotFoundError(
            f"Output pack parents do not exist: parents={missing_parents}"
        )
    existing: list[str] = [
        str(path) for path in destinations.values() if path.exists()
    ]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite public packs: paths={existing}"
        )
    return destinations


def main() -> None:
    """Export seven immutable model packs and one fleet pack."""

    args = parse_args()
    result_root: Path = cast(Path, args.result_root).resolve()
    validation_path: Path = cast(Path, args.validation_report).resolve()
    usage_path: Path = cast(Path, args.usage_summary).resolve()
    metrics_long_path: Path = cast(Path, args.metrics_long).resolve()
    metrics_summary_path: Path = cast(
        Path,
        args.metrics_summary,
    ).resolve()
    comparisons_path: Path = cast(Path, args.comparisons).resolve()
    output_root: Path = cast(Path, args.output_root).resolve()
    seven_models: tuple[str, ...] = parse_model_list(
        str(args.seven_models),
        "seven-models",
    )
    five_models: tuple[str, ...] = parse_model_list(
        str(args.five_models),
        "five-models",
    )
    if len(seven_models) != 7 or len(five_models) != 5:
        raise ValueError(
            f"Frozen support requires seven and five models: "
            f"seven={len(seven_models)}, five={len(five_models)}"
        )
    if not set(five_models).issubset(set(seven_models)):
        raise ValueError(
            f"five-models must be a subset of seven-models: "
            f"five={five_models}, seven={seven_models}"
        )
    destinations: dict[str, Path] = ensure_destinations_absent(
        output_root,
        seven_models,
    )
    validation, model_reports = validate_export_inputs(
        validation_path,
        usage_path,
        metrics_long_path,
        metrics_summary_path,
        comparisons_path,
        result_root,
        seven_models,
        five_models,
    )
    usage: JsonObject = load_json(usage_path, "usage summary")
    usage_jobs: list[JsonObject] = [
        require_object(item, f"usage.jobs[{index}]")
        for index, item in enumerate(
            require_list(usage.get("jobs"), "usage.jobs")
        )
    ]
    long_rows: list[JsonObject] = load_jsonl(
        metrics_long_path,
        "metrics long",
    )
    metrics_summary: JsonObject = load_json(
        metrics_summary_path,
        "metrics summary",
    )
    comparisons: JsonObject = load_json(
        comparisons_path,
        "paired comparisons",
    )
    validation_sha: str = sha256_file(validation_path)
    native_models: set[str] = set(five_models)
    with tempfile.TemporaryDirectory(
        prefix=".runtime-matched-baseline-export.",
        dir=output_root,
    ) as temporary_name:
        temporary_root: Path = Path(temporary_name)
        model_manifest_shas: dict[str, str] = {}
        temporary_model_dirs: dict[str, Path] = {}
        for model in seven_models:
            temporary_directory: Path = temporary_root / model
            build_model_pack(
                temporary_directory,
                result_root,
                model,
                model in native_models,
                model_reports[model],
                [job for job in usage_jobs if job.get("model") == model],
                long_rows,
                metrics_summary,
                validation_sha,
            )
            temporary_model_dirs[model] = temporary_directory
            model_manifest_shas[model] = sha256_file(
                temporary_directory / "manifest.json"
            )
        temporary_fleet: Path = temporary_root / "fleet"
        build_fleet_pack(
            temporary_fleet,
            validation,
            usage,
            long_rows,
            metrics_summary,
            comparisons,
            model_manifest_shas,
            validation_sha,
        )
        for model in seven_models:
            os.rename(temporary_model_dirs[model], destinations[model])
        os.rename(temporary_fleet, destinations["fleet"])
    print(
        canonical_json(
            {
                "event": "runtime_matched_baseline_public_export_complete",
                "model_packs": len(seven_models),
                "fleet_pack": str(destinations["fleet"]),
                "fleet_manifest_sha256": sha256_file(
                    destinations["fleet"] / "manifest.json"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
