#!/usr/bin/env python3
"""Validate one exported K-prefix cache pack before retrieval starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyskill.k_ablation import (
    DOMAINS,
    KResultSpec,
    load_cache_stamp,
    validate_cache_artifact,
)


def parse_args() -> argparse.Namespace:
    """Parse explicit cache identity and dataset arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--instances-dir", required=True, type=Path)
    parser.add_argument("--domains", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", required=True, type=int)
    return parser.parse_args()


def parse_domains(value: str) -> tuple[str, ...]:
    """Return a unique ordered list of supported domains."""

    domains: tuple[str, ...] = tuple(part.strip() for part in value.split(",") if part.strip())
    if not domains:
        raise ValueError("At least one cache-validation domain is required")
    if len(domains) != len(set(domains)):
        raise ValueError(f"Cache-validation domains contain duplicates: domains={domains}")
    unsupported: list[str] = sorted(set(domains) - set(DOMAINS))
    if unsupported:
        raise ValueError(
            f"Unsupported cache-validation domains: unsupported={unsupported}, allowed={DOMAINS}"
        )
    return domains


def main() -> None:
    """Validate manifest identity, source data hashes, and artifact bytes."""

    args = parse_args()
    domains: tuple[str, ...] = parse_domains(args.domains)
    repository_root: Path = Path(__file__).resolve().parents[1]
    artifact_path: Path = validate_cache_artifact(args.manifest, repository_root)
    for domain in domains:
        spec: KResultSpec = {
            "tag": args.tag,
            "model": args.model,
            "k_samples": args.k,
            "domain": domain,
            "variant": "hyskill",
            "encoder": "cache-validation-only",
            "source_revision": "cache-validation-only",
        }
        load_cache_stamp(
            args.manifest,
            args.instances_dir / f"{domain}.json",
            spec,
        )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "artifact": str(artifact_path),
                "tag": args.tag,
                "model": args.model,
                "k_samples": args.k,
                "domains": list(domains),
                "verified": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
