#!/usr/bin/env python3
"""Clone verified Bare runtime facts for one explicit Wave C job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TypeAlias, cast


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def parse_args() -> argparse.Namespace:
    """Parse required source, identity, and output arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--source-pack-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return one JSON object or raise with source context."""

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: context={context}, "
            f"value_type={type(value).__name__}"
        )
    return value


def load_json(path: Path) -> JsonObject:
    """Load one required JSON object."""

    if not path.is_file():
        raise FileNotFoundError(f"Base manifest does not exist: path={path}")
    value: JsonValue = cast(
        JsonValue,
        json.loads(path.read_text(encoding="utf-8")),
    )
    return require_object(value, "base-manifest")


def clone_runtime_facts(
    base_manifest: JsonObject,
    job_id: str,
    result_tag: str,
    model: str,
    domain: str,
    arm: str,
    stage: str,
    source_pack_sha256: str,
) -> JsonObject:
    """Return independent runtime facts for one explicit Wave C job."""

    base_facts: JsonObject = require_object(
        base_manifest.get("runtime_facts"),
        "base-manifest.runtime_facts",
    )
    facts: JsonObject = cast(
        JsonObject,
        json.loads(json.dumps(base_facts, ensure_ascii=False)),
    )
    facts["job"] = {
        "job_id": job_id,
        "result_tag": result_tag,
        "model": model,
        "domain": domain,
        "arm": arm,
        "stage": stage,
    }
    source: JsonObject = require_object(
        facts.get("source"),
        "runtime-facts.source",
    )
    source["source_pack_sha256"] = source_pack_sha256
    return facts


def write_json_atomic(path: Path, payload: JsonObject) -> None:
    """Write one formatted JSON object atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    """Clone and persist one Wave C runtime-facts document."""

    args = parse_args()
    output_path: Path = cast(Path, args.output).resolve()
    payload: JsonObject = clone_runtime_facts(
        load_json(cast(Path, args.base_manifest).resolve()),
        str(args.job_id),
        str(args.result_tag),
        str(args.model),
        str(args.domain),
        str(args.arm),
        str(args.stage),
        str(args.source_pack_sha256),
    )
    write_json_atomic(output_path, payload)
    print(str(output_path), flush=True)


if __name__ == "__main__":
    main()
