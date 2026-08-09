#!/usr/bin/env python3
"""Validate a native-BM25 artifact against frozen inputs and legacy order."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from hyskill.runtime_matched_bm25 import (
    BM25ValidationReport,
    JsonObject,
    read_json,
    require_object,
    validate_bm25_artifact,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse one exact BM25 validation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--legacy-source", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Validate all candidate, coverage, order, duplicate, and SHA gates."""

    args = parse_args()
    artifact_path: Path = cast(Path, args.artifact).resolve()
    manifest_path: Path = cast(Path, args.manifest).resolve()
    legacy_path: Path = cast(Path, args.legacy_source).resolve()
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    report_path: Path = cast(Path, args.report).resolve()
    domain: str = str(args.domain)
    artifact: JsonObject = require_object(
        read_json(artifact_path),
        "artifact",
    )
    manifest: JsonObject = require_object(
        read_json(manifest_path),
        "manifest",
    )
    legacy: JsonObject = require_object(read_json(legacy_path), "legacy")
    report: BM25ValidationReport = validate_bm25_artifact(
        artifact,
        manifest,
        legacy,
        artifact_path,
        manifest_path,
        legacy_path,
        corpus_path,
        instances_path,
        domain,
    )
    write_json(report_path, report)
    print(
        " ".join(
            (
                "runtime_matched_bm25_valid",
                f"domain={domain}",
                f"instances={report['instance_count']}",
                f"top_k={report['top_k']}",
                f"artifact_sha256={report['artifact_sha256']}",
                f"legacy_source_sha256={report['legacy_source_sha256']}",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
