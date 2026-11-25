#!/usr/bin/env python3
"""Export K=2 routed top-50 candidates into standard SR-Agents retrieval JSON.

Reads ``community-results/<TAG>/k2/retrieval_top50.jsonl.gz``, filters to one
requested domain while verifying exact order and 50-candidate uniqueness, and
writes ``{"metadata": {...}, "results": [...]}`` plus ``<output>.manifest.json``
recording ``source_sha256`` and ``output_sha256``.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import cast

from hyskill.runtime_matched_bm25 import FROZEN_TOP_K
from hyskill.runtime_matched_execution import (
    JsonObject,
    JsonValue,
    sha256_file,
    write_json_atomic,
)
from hyskill.runtime_matched_rerank import (
    RuntimeMatchedRerankError,
    require_list,
    require_object,
    require_string,
    require_supported_model,
)


def export_routed_candidates(
    source_path: Path,
    output_path: Path,
    *,
    result_tag: str,
    domain: str,
) -> dict[str, str | int]:
    """Convert one domain's K=2 routed JSONL.GZ into a standard retrieval JSON."""

    require_supported_model(result_tag)
    resolved_source: Path = source_path.resolve()
    resolved_output: Path = output_path.resolve()
    if resolved_source == resolved_output:
        raise ValueError(f"output must not overwrite source: path={resolved_source}")
    if not resolved_source.is_file():
        raise FileNotFoundError(f"routed retrieval source does not exist: path={resolved_source}")

    source_sha256: str = sha256_file(resolved_source)
    results: list[JsonObject] = []
    seen_ids: set[str] = set()

    with gzip.open(resolved_source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            context = f"{resolved_source}:{line_number}"
            raw_row = require_object(cast(JsonValue, json.loads(line)), context)
            row_domain = require_string(raw_row.get("domain"), f"{context}.domain")
            if row_domain != domain:
                continue
            row_model = require_string(raw_row.get("model"), f"{context}.model")
            if row_model != result_tag:
                raise RuntimeMatchedRerankError(
                    f"routed source model mismatch: context={context}, "
                    f"expected={result_tag}, actual={row_model}"
                )
            variant = require_string(raw_row.get("variant"), f"{context}.variant")
            if variant != "routed":
                raise RuntimeMatchedRerankError(
                    f"routed source variant is not 'routed': context={context}, actual={variant}"
                )
            if raw_row.get("k_samples") != 2:
                raise RuntimeMatchedRerankError(
                    f"routed source k_samples is not 2: context={context}, "
                    f"actual={raw_row.get('k_samples')!r}"
                )
            instance_id = require_string(raw_row.get("instance_id"), f"{context}.instance_id")
            if instance_id in seen_ids:
                raise RuntimeMatchedRerankError(
                    f"duplicate instance_id in routed source: {instance_id}"
                )
            seen_ids.add(instance_id)

            gold_list = [
                require_string(gid, f"{context}.gold_skill_ids[{idx}]")
                for idx, gid in enumerate(
                    require_list(raw_row.get("gold_skill_ids"), f"{context}.gold_skill_ids")
                )
            ]
            retrieved_raw = require_list(raw_row.get("retrieved"), f"{context}.retrieved")
            if len(retrieved_raw) != FROZEN_TOP_K:
                raise RuntimeMatchedRerankError(
                    f"routed source does not have {FROZEN_TOP_K} candidates: "
                    f"context={context}, instance_id={instance_id}, actual={len(retrieved_raw)}"
                )
            retrieved_items: list[JsonObject] = []
            seen_skills: set[str] = set()
            for rank, item in enumerate(retrieved_raw):
                obj = require_object(item, f"{context}.retrieved[{rank}]")
                sid = require_string(obj.get("skill_id"), f"{context}.retrieved[{rank}].skill_id")
                if sid in seen_skills:
                    raise RuntimeMatchedRerankError(
                        f"duplicate candidate skill_id={sid} in {context}"
                    )
                seen_skills.add(sid)
                score_val = obj.get("score")
                if not isinstance(score_val, (int, float)) or isinstance(score_val, bool):
                    raise RuntimeMatchedRerankError(
                        f"candidate score must be numeric: {context}.retrieved[{rank}].score"
                    )
                retrieved_items.append({"skill_id": sid, "score": float(score_val)})

            results.append(
                {
                    "instance_id": instance_id,
                    "gold_skill_ids": gold_list,
                    "retrieved": retrieved_items,
                }
            )

    if not results:
        raise RuntimeMatchedRerankError(
            f"no routed rows found for domain={domain} in {resolved_source}"
        )

    payload: JsonObject = {
        "metadata": {
            "schema_version": "ijcai27-routed-candidates-v1",
            "result_tag": result_tag,
            "domain": domain,
            "variant": "routed",
            "k_samples": 2,
            "top_k": FROZEN_TOP_K,
            "source_sha256": source_sha256,
            "instance_count": len(results),
        },
        "results": results,
    }
    write_json_atomic(resolved_output, payload)
    output_sha256: str = sha256_file(resolved_output)
    manifest_path = Path(str(resolved_output) + ".manifest.json")
    manifest: JsonObject = {
        "schema_version": "ijcai27-routed-candidates-manifest-v1",
        "result_tag": result_tag,
        "domain": domain,
        "instance_count": len(results),
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "output_bytes": resolved_output.stat().st_size,
    }
    write_json_atomic(manifest_path, manifest)
    return {
        "result_tag": result_tag,
        "domain": domain,
        "instance_count": len(results),
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export K=2 routed top-50 candidates for hy_rerank."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--domain", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_routed_candidates(
        args.source,
        args.output,
        result_tag=str(args.result_tag),
        domain=str(args.domain),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
