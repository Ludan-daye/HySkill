#!/usr/bin/env python3
"""Validate and stamp one raw SR-Agents K-ablation retrieval result."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from hyskill.k_ablation import KResultSpec, finalize_result_payload, load_json_object


def parse_args() -> argparse.Namespace:
    """Parse explicit result identity and file arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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


def write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    """Atomically create a JSON file without overwriting an existing result."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result: path={path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
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
    """Finalize one result and print its validated identity."""

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
    finalized: dict[str, object] = finalize_result_payload(
        load_json_object(args.input), args.instances, args.cache_manifest, spec
    )
    write_json_exclusive(args.output, finalized)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tag": args.tag,
                "model": args.model,
                "k_samples": args.k,
                "domain": args.domain,
                "variant": args.variant,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
