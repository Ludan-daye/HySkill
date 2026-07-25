#!/usr/bin/env python3
"""Build one deterministic native-BM25 top-50 artifact and manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from hyskill.runtime_matched_bm25 import (
    FROZEN_SRAGENTS_REVISION,
    FROZEN_TOP_K,
    BM25Artifact,
    BM25Manifest,
    JsonObject,
    JsonValue,
    build_bm25_artifact,
    build_bm25_manifest,
    load_native_bm25_runtime,
    read_json,
    require_list,
    sha256_file,
    validated_corpus,
    validated_instances,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse one explicit domain build."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--sragents-checkout", required=True, type=Path)
    parser.add_argument(
        "--sragents-revision",
        required=True,
        choices=(FROZEN_SRAGENTS_REVISION,),
    )
    return parser.parse_args()


def main() -> None:
    """Build, hash, and write one frozen BM25 candidate source."""

    args = parse_args()
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    manifest_path: Path = cast(Path, args.manifest).resolve()
    domain: str = str(args.domain)
    checkout_path: Path = cast(Path, args.sragents_checkout).resolve()
    revision: str = str(args.sragents_revision)
    if output_path == manifest_path:
        raise ValueError(
            f"Output and manifest paths must differ: path={output_path}"
        )

    corpus_values: list[JsonValue] = require_list(
        read_json(corpus_path),
        "corpus",
    )
    instance_values: list[JsonValue] = require_list(
        read_json(instances_path),
        "instances",
    )
    corpus: list[JsonObject] = validated_corpus(corpus_values)
    instances: list[JsonObject] = validated_instances(instance_values, domain)
    runtime = load_native_bm25_runtime(checkout_path, revision)
    artifact: BM25Artifact = build_bm25_artifact(
        instances,
        corpus,
        domain,
        runtime,
        sha256_file(corpus_path),
        sha256_file(instances_path),
        FROZEN_TOP_K,
    )
    write_json(output_path, artifact)
    manifest: BM25Manifest = build_bm25_manifest(
        artifact,
        output_path,
        corpus_path,
        instances_path,
        runtime,
        (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1]
            / "hyskill"
            / "runtime_matched_bm25.py",
            Path(__file__).resolve().parents[1]
            / "hyskill"
            / "bm25.py",
        ),
    )
    write_json(manifest_path, manifest)
    print(
        " ".join(
            (
                "runtime_matched_bm25_complete",
                f"domain={domain}",
                f"instances={len(instances)}",
                f"top_k={FROZEN_TOP_K}",
                f"artifact_sha256={manifest['artifact_sha256']}",
                "legacy_order_validation=pending",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
