#!/usr/bin/env python3
"""Compare K=2 HySkill arms with K-independent answer baselines."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    JsonObject,
    JsonValue,
    canonical_json,
    sha256_file,
)
from scripts.audit_k2_reuse import (
    require_list,
    require_object,
    require_string,
)
from scripts.summarize_k2_answer_evaluations import (
    DOMAIN_COUNTS,
    DOMAIN_VALIDATION_COUNTS,
    DOMAINS,
    AnswerEvaluationRow,
    ComparisonSpec,
    load_rows,
    paired_report,
    parse_model_list,
    verify_coverage,
    write_json_atomic,
)


QWEN4_RESULT_TAG: str = "qwen3.5-4b-reference"
QWEN4_LEGACY_MODEL: str = "qwen35-4b"
BASELINE_CAVEAT: str = (
    "The compact baseline sources preserve per-instance correctness and file "
    "identity, but they do not prove checkpoint, tokenizer, chat-template, or "
    "runtime equivalence with the K=2 endpoints. These comparisons may be "
    "reported only after the separate K2M001 runtime-identity gate passes."
)


class ComparisonRow(TypedDict):
    """One correctness row accepted by the shared paired-bootstrap engine."""

    model: str
    domain: str
    arm: str
    instance_id: str
    correct: bool
    failure_category: str
    is_validation: bool


class BaselineComparisonSpec(TypedDict):
    """One registered K=2 versus K-independent baseline comparison."""

    name: str
    arm_a: str
    arm_b: str
    models: tuple[str, ...]


ReferenceKey = tuple[str, str, str]
DomainInstanceKey = tuple[str, str]


def parse_args() -> argparse.Namespace:
    """Parse explicit K=2 support, baseline source, and output arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--k2-evals", required=True, nargs="+", type=Path)
    parser.add_argument("--community-root", required=True, type=Path)
    parser.add_argument("--seven-models", required=True)
    parser.add_argument("--five-models", required=True)
    parser.add_argument("--fixed-model", required=True)
    parser.add_argument("--expected-eval-files", required=True, type=int)
    parser.add_argument("--expected-total-per-model", required=True, type=int)
    parser.add_argument("--expected-heldout-per-model", required=True, type=int)
    parser.add_argument("--bootstrap-samples", required=True, type=int)
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_boolean(value: JsonValue | None, context: str) -> bool:
    """Return one JSON Boolean with source context."""

    if not isinstance(value, bool):
        raise DownstreamDataError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


def require_integer(value: JsonValue | None, context: str) -> int:
    """Return one JSON integer with source context."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise DownstreamDataError(
            f"Expected integer: context={context}, value={value!r}"
        )
    return value


def protocol_support(
    seven_models_value: str,
    five_models_value: str,
    fixed_model: str,
    expected_total_per_model: int,
    expected_heldout_per_model: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the frozen model and instance support."""

    seven_models: tuple[str, ...] = parse_model_list(
        seven_models_value,
        "seven-models",
    )
    five_models: tuple[str, ...] = parse_model_list(
        five_models_value,
        "five-models",
    )
    if len(seven_models) != 7:
        raise ValueError(
            "seven-models must contain exactly 7 models: "
            f"actual={len(seven_models)}"
        )
    if len(five_models) != 5:
        raise ValueError(
            "five-models must contain exactly 5 models: "
            f"actual={len(five_models)}"
        )
    if not set(five_models).issubset(set(seven_models)):
        raise ValueError(
            "five-models must be a subset of seven-models: "
            f"five={five_models}, seven={seven_models}"
        )
    if fixed_model not in set(five_models):
        raise ValueError(
            f"fixed-model must belong to five-models: model={fixed_model}"
        )
    if fixed_model != QWEN4_RESULT_TAG:
        raise ValueError(
            "The explicit Qwen4 baseline adapter requires the frozen fixed "
            f"model tag: expected={QWEN4_RESULT_TAG}, actual={fixed_model}"
        )
    protocol_total: int = sum(DOMAIN_COUNTS.values())
    protocol_heldout: int = protocol_total - sum(
        DOMAIN_VALIDATION_COUNTS.values()
    )
    if expected_total_per_model != protocol_total:
        raise ValueError(
            "Explicit total denominator disagrees with the frozen protocol: "
            f"explicit={expected_total_per_model}, protocol={protocol_total}"
        )
    if expected_heldout_per_model != protocol_heldout:
        raise ValueError(
            "Explicit held-out denominator disagrees with the frozen protocol: "
            f"explicit={expected_heldout_per_model}, protocol={protocol_heldout}"
        )
    return seven_models, five_models


def resolved_unique_paths(
    paths: Sequence[Path],
    expected_count: int,
) -> list[Path]:
    """Resolve an exact, duplicate-free K=2 evaluation file set."""

    resolved: list[Path] = [path.resolve() for path in paths]
    if len(resolved) != expected_count:
        raise DownstreamDataError(
            "K=2 evaluation file count mismatch: "
            f"expected={expected_count}, actual={len(resolved)}"
        )
    if len(resolved) != len(set(resolved)):
        raise DownstreamDataError(
            "K=2 evaluation inputs contain duplicate paths"
        )
    return resolved


def k2_validation_reference(
    rows: Sequence[AnswerEvaluationRow],
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> dict[ReferenceKey, bool]:
    """Use K=2 routed-Gated rows as the only held-out split authority."""

    seven_model_set: set[str] = set(seven_models)
    five_model_set: set[str] = set(five_models)
    reference: dict[ReferenceKey, bool] = {}
    for row in rows:
        if row["model"] not in seven_model_set:
            continue
        if row["arm"] != "routed_gated":
            continue
        key: ReferenceKey = (
            row["model"],
            row["domain"],
            row["instance_id"],
        )
        if key in reference:
            raise DownstreamDataError(
                f"Duplicate K=2 routed-Gated split reference: key={key}"
            )
        reference[key] = row["is_validation"]
    expected_reference_count: int = len(seven_models) * sum(
        DOMAIN_COUNTS.values()
    )
    if len(reference) != expected_reference_count:
        raise DownstreamDataError(
            "K=2 routed-Gated split reference is incomplete: "
            f"expected={expected_reference_count}, actual={len(reference)}"
        )
    for row in rows:
        if row["model"] not in five_model_set:
            continue
        if row["arm"] != "routed_select":
            continue
        key = (row["model"], row["domain"], row["instance_id"])
        expected_validation: bool | None = reference.get(key)
        if expected_validation is None:
            raise DownstreamDataError(
                f"K=2 Select row is outside routed-Gated support: key={key}"
            )
        if row["is_validation"] != expected_validation:
            raise DownstreamDataError(
                "K=2 Select and routed-Gated split flags disagree: "
                f"key={key}, gated={expected_validation}, "
                f"select={row['is_validation']}"
            )
    return reference


def load_gzip_jsonl(path: Path) -> list[JsonObject]:
    """Load one gzip JSONL baseline source without schema fallback."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Baseline compact source does not exist: path={path}"
        )
    rows: list[JsonObject] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    raw_row: JsonValue = cast(JsonValue, json.loads(line))
                except json.JSONDecodeError as error:
                    raise DownstreamDataError(
                        "Baseline JSONL is malformed: "
                        f"path={path}, line={line_number}, "
                        f"column={error.colno}, message={error.msg}"
                    ) from error
                rows.append(
                    require_object(
                        raw_row,
                        f"baseline:{path}:{line_number}",
                    )
                )
    except gzip.BadGzipFile as error:
        raise DownstreamDataError(
            f"Baseline source is not valid gzip: path={path}"
        ) from error
    if not rows:
        raise DownstreamDataError(
            f"Baseline compact source is empty: path={path}"
        )
    return rows


def baseline_row(
    model: str,
    domain: str,
    arm: str,
    instance_id: str,
    correct: bool,
    is_validation: bool,
) -> ComparisonRow:
    """Build one baseline row using the K=2 split flag."""

    return {
        "model": model,
        "domain": domain,
        "arm": arm,
        "instance_id": instance_id,
        "correct": correct,
        "failure_category": "legacy_compact_correctness",
        "is_validation": is_validation,
    }


def legacy_validation_field(model: str) -> str:
    """Return the explicit legacy split field for one compact-pack schema."""

    if model == QWEN4_RESULT_TAG:
        return "in_calibration_split_routed"
    return "in_calibration_split"


def expected_model_keys(
    reference: Mapping[ReferenceKey, bool],
    model: str,
) -> set[DomainInstanceKey]:
    """Return exact K=2 domain-instance support for one model."""

    return {
        (domain, instance_id)
        for reference_model, domain, instance_id in reference
        if reference_model == model
    }


def load_compact_model_baselines(
    community_root: Path,
    model: str,
    supports_native_baselines: bool,
    reference: Mapping[ReferenceKey, bool],
) -> tuple[list[ComparisonRow], JsonObject]:
    """Load one model's compact correctness fields with exact ID coverage."""

    path: Path = (
        community_root / model / "k4" / "gating_per_instance.jsonl.gz"
    )
    raw_rows: list[JsonObject] = load_gzip_jsonl(path)
    expected_keys: set[DomainInstanceKey] = expected_model_keys(
        reference,
        model,
    )
    observed_keys: set[DomainInstanceKey] = set()
    output: list[ComparisonRow] = []
    split_field: str = legacy_validation_field(model)
    split_mismatches: int = 0
    for row_index, row in enumerate(raw_rows):
        context: str = f"baseline:{path}:{row_index + 1}"
        instance_id: str = require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        )
        domain: str = require_string(
            row.get("domain"),
            f"{context}.domain",
        )
        if domain not in DOMAINS:
            raise DownstreamDataError(
                f"Unknown baseline domain: context={context}, domain={domain}"
            )
        key: DomainInstanceKey = (domain, instance_id)
        if key in observed_keys:
            raise DownstreamDataError(
                f"Duplicate baseline compact row: model={model}, key={key}"
            )
        observed_keys.add(key)
        reference_key: ReferenceKey = (model, domain, instance_id)
        if reference_key not in reference:
            raise DownstreamDataError(
                "Baseline compact row is outside K=2 support: "
                f"model={model}, key={key}"
            )
        is_validation: bool = reference[reference_key]
        legacy_validation: bool = require_boolean(
            row.get(split_field),
            f"{context}.{split_field}",
        )
        if legacy_validation != is_validation:
            split_mismatches += 1
        output.append(
            baseline_row(
                model,
                domain,
                "bare",
                instance_id,
                require_boolean(
                    row.get("correct_bare"),
                    f"{context}.correct_bare",
                ),
                is_validation,
            )
        )
        if supports_native_baselines and model != QWEN4_RESULT_TAG:
            output.append(
                baseline_row(
                    model,
                    domain,
                    "always_rerank",
                    instance_id,
                    require_boolean(
                        row.get("correct_always_rerank"),
                        f"{context}.correct_always_rerank",
                    ),
                    is_validation,
                )
            )
            output.append(
                baseline_row(
                    model,
                    domain,
                    "select_bm25",
                    instance_id,
                    require_boolean(
                        row.get("correct_select_bm25"),
                        f"{context}.correct_select_bm25",
                    ),
                    is_validation,
                )
            )
    if observed_keys != expected_keys:
        raise DownstreamDataError(
            "Baseline compact support mismatch: "
            f"model={model}, missing={sorted(expected_keys - observed_keys)[:10]}, "
            f"unexpected={sorted(observed_keys - expected_keys)[:10]}"
        )
    if split_mismatches:
        raise DownstreamDataError(
            "Legacy and K=2 validation IDs disagree: "
            f"model={model}, field={split_field}, mismatches={split_mismatches}"
        )
    provenance: JsonObject = {
        "adapter": (
            "qwen4_compact_bare_only"
            if model == QWEN4_RESULT_TAG
            else (
                "compact_bare_and_native_fields"
                if supports_native_baselines
                else "compact_bare_only"
            )
        ),
        "gating_per_instance": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        },
        "legacy_validation_field": split_field,
        "legacy_validation_matches_k2": True,
    }
    return output, provenance


def load_json_object(path: Path) -> JsonObject:
    """Load one JSON object with actionable source context."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Qwen4 native baseline source does not exist: path={path}"
        )
    try:
        raw_value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise DownstreamDataError(
            "Qwen4 native baseline JSON is malformed: "
            f"path={path}, line={error.lineno}, column={error.colno}, "
            f"message={error.msg}"
        ) from error
    return require_object(raw_value, f"qwen4-native:{path}")


def load_qwen4_native_arm(
    community_root: Path,
    arm: str,
    reference: Mapping[ReferenceKey, bool],
) -> tuple[list[ComparisonRow], list[JsonObject]]:
    """Load one Qwen4 native arm from its explicit legacy schema."""

    output: list[ComparisonRow] = []
    provenance: list[JsonObject] = []
    model_expected_keys: set[DomainInstanceKey] = expected_model_keys(
        reference,
        QWEN4_RESULT_TAG,
    )
    observed_keys: set[DomainInstanceKey] = set()
    for domain in DOMAINS:
        path: Path = (
            community_root
            / QWEN4_RESULT_TAG
            / "baselines-native"
            / f"{domain}-{arm}.eval.json"
        )
        payload: JsonObject = load_json_object(path)
        if payload.get("dataset") != domain:
            raise DownstreamDataError(
                "Qwen4 native dataset mismatch: "
                f"path={path}, expected={domain}, actual={payload.get('dataset')!r}"
            )
        if payload.get("method") != arm:
            raise DownstreamDataError(
                "Qwen4 native method mismatch: "
                f"path={path}, expected={arm}, actual={payload.get('method')!r}"
            )
        if payload.get("model") != QWEN4_LEGACY_MODEL:
            raise DownstreamDataError(
                "Qwen4 native model alias mismatch: "
                f"path={path}, expected={QWEN4_LEGACY_MODEL}, "
                f"actual={payload.get('model')!r}"
            )
        metrics: JsonObject = require_object(
            payload.get("metrics"),
            f"qwen4-native:{path}.metrics",
        )
        expected_domain_count: int = DOMAIN_COUNTS[domain]
        if (
            require_integer(
                metrics.get("total"),
                f"qwen4-native:{path}.metrics.total",
            )
            != expected_domain_count
        ):
            raise DownstreamDataError(
                "Qwen4 native metric denominator mismatch: "
                f"path={path}, expected={expected_domain_count}, "
                f"actual={metrics.get('total')!r}"
            )
        details: list[JsonValue] = require_list(
            payload.get("details"),
            f"qwen4-native:{path}.details",
        )
        if len(details) != expected_domain_count:
            raise DownstreamDataError(
                "Qwen4 native detail denominator mismatch: "
                f"path={path}, expected={expected_domain_count}, "
                f"actual={len(details)}"
            )
        for row_index, raw_row in enumerate(details):
            row: JsonObject = require_object(
                raw_row,
                f"qwen4-native:{path}.details[{row_index}]",
            )
            instance_id: str = require_string(
                row.get("instance_id"),
                f"qwen4-native:{path}.details[{row_index}].instance_id",
            )
            key: DomainInstanceKey = (domain, instance_id)
            if key in observed_keys:
                raise DownstreamDataError(
                    f"Duplicate Qwen4 native baseline row: arm={arm}, key={key}"
                )
            observed_keys.add(key)
            reference_key: ReferenceKey = (
                QWEN4_RESULT_TAG,
                domain,
                instance_id,
            )
            if reference_key not in reference:
                raise DownstreamDataError(
                    "Qwen4 native row is outside K=2 support: "
                    f"arm={arm}, key={key}"
                )
            output.append(
                baseline_row(
                    QWEN4_RESULT_TAG,
                    domain,
                    arm,
                    instance_id,
                    require_boolean(
                        row.get("correct"),
                        f"qwen4-native:{path}.details[{row_index}].correct",
                    ),
                    reference[reference_key],
                )
            )
        provenance.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "domain": domain,
                "arm": arm,
                "legacy_model_alias": QWEN4_LEGACY_MODEL,
            }
        )
    if observed_keys != model_expected_keys:
        raise DownstreamDataError(
            "Qwen4 native support mismatch: "
            f"arm={arm}, missing={sorted(model_expected_keys - observed_keys)[:10]}, "
            f"unexpected={sorted(observed_keys - model_expected_keys)[:10]}"
        )
    return output, provenance


def load_baseline_rows(
    community_root: Path,
    seven_models: Sequence[str],
    five_models: Sequence[str],
    reference: Mapping[ReferenceKey, bool],
) -> tuple[list[ComparisonRow], JsonObject]:
    """Load all seven-model Bare and five-model native baseline rows."""

    output: list[ComparisonRow] = []
    provenance: JsonObject = {}
    five_model_set: set[str] = set(five_models)
    for model in seven_models:
        model_rows, model_provenance = load_compact_model_baselines(
            community_root,
            model,
            model in five_model_set,
            reference,
        )
        output.extend(model_rows)
        provenance[model] = model_provenance
    qwen_rerank_rows, qwen_rerank_provenance = load_qwen4_native_arm(
        community_root,
        "always_rerank",
        reference,
    )
    qwen_select_rows, qwen_select_provenance = load_qwen4_native_arm(
        community_root,
        "select_bm25",
        reference,
    )
    output.extend(qwen_rerank_rows)
    output.extend(qwen_select_rows)
    qwen_provenance: JsonObject = require_object(
        provenance.get(QWEN4_RESULT_TAG),
        f"provenance.{QWEN4_RESULT_TAG}",
    )
    qwen_provenance["native_eval_files"] = [
        *qwen_rerank_provenance,
        *qwen_select_provenance,
    ]
    qwen_provenance["adapter"] = "qwen4_compact_bare_plus_native_eval_files"
    return output, provenance


def comparison_rows(
    k2_rows: Sequence[AnswerEvaluationRow],
    baseline_rows: Sequence[ComparisonRow],
) -> list[ComparisonRow]:
    """Keep only K=2 arms needed by the four baseline comparisons."""

    output: list[ComparisonRow] = [
        {
            "model": row["model"],
            "domain": row["domain"],
            "arm": row["arm"],
            "instance_id": row["instance_id"],
            "correct": row["correct"],
            "failure_category": row["failure_category"],
            "is_validation": row["is_validation"],
        }
        for row in k2_rows
        if row["arm"] in ("routed_gated", "routed_select")
    ]
    output.extend(baseline_rows)
    return output


def comparison_specs(
    seven_models: tuple[str, ...],
    five_models: tuple[str, ...],
) -> tuple[BaselineComparisonSpec, ...]:
    """Return the four missing preregistered baseline comparisons."""

    return (
        {
            "name": "gated_vs_bare_seven_model",
            "arm_a": "routed_gated",
            "arm_b": "bare",
            "models": seven_models,
        },
        {
            "name": "gated_vs_native_rerank_five_model",
            "arm_a": "routed_gated",
            "arm_b": "always_rerank",
            "models": five_models,
        },
        {
            "name": "gated_vs_native_select_five_model",
            "arm_a": "routed_gated",
            "arm_b": "select_bm25",
            "models": five_models,
        },
        {
            "name": "hyskill_select_vs_bm25_select_five_model",
            "arm_a": "routed_select",
            "arm_b": "select_bm25",
            "models": five_models,
        },
    )


def shared_paired_report(
    rows: Sequence[ComparisonRow],
    specification: BaselineComparisonSpec,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> JsonObject:
    """Call the existing paired and hierarchical bootstrap implementation."""

    runtime_rows: Sequence[AnswerEvaluationRow] = cast(
        Sequence[AnswerEvaluationRow],
        rows,
    )
    runtime_specification: ComparisonSpec = cast(
        ComparisonSpec,
        specification,
    )
    return paired_report(
        runtime_rows,
        runtime_specification,
        bootstrap_samples,
        bootstrap_seed,
    )


def main() -> None:
    """Validate exact support and write the four baseline comparisons."""

    args = parse_args()
    expected_eval_files: int = int(args.expected_eval_files)
    expected_total_per_model: int = int(args.expected_total_per_model)
    expected_heldout_per_model: int = int(args.expected_heldout_per_model)
    bootstrap_samples: int = int(args.bootstrap_samples)
    bootstrap_seed: int = int(args.bootstrap_seed)
    if expected_eval_files <= 0:
        raise ValueError(
            f"expected-eval-files must be positive: value={expected_eval_files}"
        )
    if bootstrap_samples <= 0:
        raise ValueError(
            f"bootstrap-samples must be positive: value={bootstrap_samples}"
        )
    fixed_model: str = str(args.fixed_model)
    seven_models, five_models = protocol_support(
        str(args.seven_models),
        str(args.five_models),
        fixed_model,
        expected_total_per_model,
        expected_heldout_per_model,
    )
    k2_paths: list[Path] = resolved_unique_paths(
        cast(list[Path], args.k2_evals),
        expected_eval_files,
    )
    k2_rows: list[AnswerEvaluationRow] = load_rows(k2_paths)
    verify_coverage(
        k2_rows,
        seven_models,
        five_models,
        fixed_model,
        expected_total_per_model,
        expected_heldout_per_model,
    )
    reference: dict[ReferenceKey, bool] = k2_validation_reference(
        k2_rows,
        seven_models,
        five_models,
    )
    community_root: Path = cast(Path, args.community_root).resolve()
    if not community_root.is_dir():
        raise NotADirectoryError(
            f"Community result root does not exist: path={community_root}"
        )
    baseline_rows, baseline_provenance = load_baseline_rows(
        community_root,
        seven_models,
        five_models,
        reference,
    )
    paired_rows: list[ComparisonRow] = comparison_rows(
        k2_rows,
        baseline_rows,
    )
    specifications: tuple[BaselineComparisonSpec, ...] = comparison_specs(
        seven_models,
        five_models,
    )
    comparisons: JsonObject = {
        specification["name"]: shared_paired_report(
            paired_rows,
            specification,
            bootstrap_samples,
            bootstrap_seed,
        )
        for specification in specifications
    }
    payload: JsonObject = {
        "schema_version": "k2-baseline-paired-comparisons-v1",
        "split": "heldout",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "comparison_count": len(specifications),
        "comparisons": comparisons,
        "support": {
            "seven_models": list(seven_models),
            "five_models": list(five_models),
            "expected_total_per_model": expected_total_per_model,
            "expected_heldout_per_model": expected_heldout_per_model,
            "k2_evaluation_files": len(k2_paths),
            "k2_evaluation_rows": len(k2_rows),
            "baseline_rows": len(baseline_rows),
        },
        "provenance": {
            "k2_evaluations": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in sorted(k2_paths)
            ],
            "baseline_sources": baseline_provenance,
            "heldout_authority": (
                "K=2 routed_gated is_validation flags; legacy split fields "
                "were checked for exact agreement but were not used."
            ),
            "baseline_runtime_identity_gate": "not_proven_by_this_script",
            "caveat": BASELINE_CAVEAT,
        },
    }
    output_path: Path = cast(Path, args.output).resolve()
    write_json_atomic(output_path, payload)
    print(
        canonical_json(
            {
                "event": "k2_baseline_paired_comparisons_complete",
                "comparison_count": len(specifications),
                "k2_evaluation_files": len(k2_paths),
                "k2_evaluation_rows": len(k2_rows),
                "baseline_rows": len(baseline_rows),
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            }
        )
    )


if __name__ == "__main__":
    main()
