#!/usr/bin/env python3
"""Validate one finalized K-ablation retrieval result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyskill.k_ablation import KResultSpec, load_json_object, validate_final_result


def parse_args() -> argparse.Namespace:
    """Parse explicit result identity and file arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--cache-manifest", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def main() -> None:
    """Validate one result and print a compact success record."""

    args = parse_args()
    spec: KResultSpec = {
        "tag": args.tag,
        "model": args.model,
        "k_samples": args.k,
        "domain": args.domain,
        "variant": args.variant,
        "encoder": args.encoder,
        "source_revision": args.source_revision,
    }
    payload = validate_final_result(
        load_json_object(args.result), args.instances, args.cache_manifest, spec
    )
    print(
        json.dumps(
            {
                "result": str(args.result),
                "k_samples": args.k,
                "domain": args.domain,
                "variant": args.variant,
                "records": len(payload["results"]),
                "verified": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
