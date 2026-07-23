#!/usr/bin/env python3
"""Build one strictly validated routed result for the K ablation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

from hyskill.k_ablation import (
    FIXED_VARIANTS,
    KResultSpec,
    RetrievalPayload,
    compute_metrics,
    expected_router_pick,
    finalize_result_payload,
    instance_ids_sha256,
    load_json_object,
    sha256_file,
    validate_final_result,
    validation_ids,
)


def parse_args() -> argparse.Namespace:
    """Parse explicit matrix identity and output arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--cache-manifest", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def result_spec(args: argparse.Namespace, variant: str) -> KResultSpec:
    """Return one fully explicit result identity."""

    return {
        "tag": args.tag,
        "model": args.model,
        "k_samples": args.k,
        "domain": args.domain,
        "variant": variant,
        "encoder": args.encoder,
        "source_revision": args.source_revision,
    }


def load_fixed_results(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, object]], dict[str, RetrievalPayload]]:
    """Load and validate the complete five-variant source matrix."""

    raw_results: dict[str, dict[str, object]] = {}
    payloads: dict[str, RetrievalPayload] = {}
    expected_ids: frozenset[str] | None = None
    for variant in FIXED_VARIANTS:
        path: Path = args.input_dir / f"{args.domain}-{variant}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Required fixed result is missing: domain={args.domain}, "
                f"variant={variant}, path={path}"
            )
        value: dict[str, object] = load_json_object(path)
        payload: RetrievalPayload = validate_final_result(
            value,
            args.instances,
            args.cache_manifest,
            result_spec(args, variant),
        )
        actual_ids: frozenset[str] = frozenset(
            record["instance_id"] for record in payload["results"]
        )
        if expected_ids is None:
            expected_ids = actual_ids
        elif actual_ids != expected_ids:
            raise ValueError(
                f"Fixed variants have different instance coverage: domain={args.domain}, "
                f"variant={variant}"
            )
        raw_results[variant] = value
        payloads[variant] = payload
    return raw_results, payloads


def route_payload(
    args: argparse.Namespace,
    raw_results: dict[str, dict[str, object]],
    payloads: dict[str, RetrievalPayload],
) -> tuple[dict[str, object], str, dict[str, float]]:
    """Select the validation winner and return a stamped routed payload."""

    first_payload: RetrievalPayload = payloads[FIXED_VARIANTS[0]]
    all_ids: frozenset[str] = frozenset(
        record["instance_id"] for record in first_payload["results"]
    )
    selected_validation_ids: frozenset[str] = validation_ids(all_ids)
    scores: dict[str, float] = {
        variant: compute_metrics(payloads[variant]["results"], selected_validation_ids)[
            "nDCG@10"
        ]
        for variant in FIXED_VARIANTS
    }
    pick, degenerate = expected_router_pick(scores)
    source_path: Path = args.input_dir / f"{args.domain}-{pick}.json"
    routed_raw: dict[str, object] = copy.deepcopy(raw_results[pick])
    metadata_value: object = routed_raw.get("metadata")
    if not isinstance(metadata_value, dict):
        raise TypeError(
            f"Selected source metadata must be an object: path={source_path}, "
            f"actual={type(metadata_value).__name__}"
        )
    metadata: dict[str, object] = metadata_value
    metadata["router"] = {
        "schema_version": 1,
        "pick": pick,
        "validation_metric": "nDCG@10",
        "validation_scores": scores,
        "n_validation": len(selected_validation_ids),
        "validation_fraction": 0.2,
        "seed": 0,
        "validation_ids_sha256": instance_ids_sha256(selected_validation_ids),
        "degenerate": degenerate,
        "source_result": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
    }
    finalized: dict[str, object] = finalize_result_payload(
        routed_raw,
        args.instances,
        args.cache_manifest,
        result_spec(args, "routed"),
    )
    validate_final_result(
        finalized,
        args.instances,
        args.cache_manifest,
        result_spec(args, "routed"),
    )
    return finalized, pick, scores


def write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    """Atomically create a routed result without overwriting prior evidence."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing routed result: path={path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path: Path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, mode="w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def main() -> None:
    """Validate five sources, route on validation, and write one result."""

    args = parse_args()
    raw_results, payloads = load_fixed_results(args)
    finalized, pick, scores = route_payload(args, raw_results, payloads)
    write_json_exclusive(args.output, finalized)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tag": args.tag,
                "model": args.model,
                "k_samples": args.k,
                "domain": args.domain,
                "pick": pick,
                "validation_scores": scores,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
