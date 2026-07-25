#!/usr/bin/env python3
"""Build verified provenance manifests for the completed K=2 answer fleet."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import cast

from hyskill.downstream_reuse import (
    CodeFileDigest,
    JsonObject,
    canonical_json,
    code_bundle_sha256_from_digests,
    code_file_digests,
    sha256_bytes,
)
from hyskill.k2_answer_provenance import (
    PRODUCTION_CONTRACT,
    build_fleet_payloads,
    json_bytes,
)


def parse_args() -> argparse.Namespace:
    """Parse the fully explicit preservation and evidence inputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument(
        "--raw-sources",
        required=True,
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--qwen-run-logs",
        required=True,
        nargs="+",
        type=Path,
    )
    return parser.parse_args()


def write_payloads_new_atomic(
    output_root: Path,
    payloads: dict[str, JsonObject],
) -> dict[str, str]:
    """Publish a validated manifest set without overwriting existing files."""

    if not output_root.is_dir():
        raise NotADirectoryError(
            f"Evidence output root does not exist: path={output_root}"
        )
    targets: dict[str, Path] = {
        filename: output_root / filename for filename in payloads
    }
    existing: list[Path] = sorted(
        path for path in targets.values() if path.exists()
    )
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing provenance manifests: "
            f"paths={[str(path) for path in existing]}"
        )
    temporary_paths: dict[str, Path] = {}
    published_paths: list[Path] = []
    output_hashes: dict[str, str] = {}
    try:
        for filename, payload in payloads.items():
            content: bytes = json_bytes(payload)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_root,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                output_file.write(content)
                output_file.flush()
                os.fsync(output_file.fileno())
                temporary_paths[filename] = Path(output_file.name)
            output_hashes[filename] = sha256_bytes(content)
        for filename in sorted(targets):
            temporary_path: Path = temporary_paths[filename]
            target_path: Path = targets[filename]
            os.link(temporary_path, target_path)
            published_paths.append(target_path)
        for temporary_path in temporary_paths.values():
            temporary_path.unlink()
        return output_hashes
    except BaseException:
        for target_path in published_paths:
            target_path.unlink(missing_ok=True)
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    """Validate all raw-to-final bindings and publish the manifests."""

    args = parse_args()
    repository_root: Path = cast(Path, args.repository_root).resolve()
    evidence_root: Path = cast(Path, args.evidence_root).resolve()
    formal_root: Path = cast(Path, args.formal_root).resolve()
    raw_sources: list[Path] = [
        path.resolve() for path in cast(list[Path], args.raw_sources)
    ]
    qwen_run_logs: list[Path] = [
        path.resolve() for path in cast(list[Path], args.qwen_run_logs)
    ]
    if not repository_root.is_dir():
        raise NotADirectoryError(
            f"Repository root does not exist: path={repository_root}"
        )
    if not evidence_root.is_dir():
        raise NotADirectoryError(
            f"Evidence root does not exist: path={evidence_root}"
        )
    if not formal_root.is_dir():
        raise NotADirectoryError(
            f"Formal root does not exist: path={formal_root}"
        )
    code_paths: list[Path] = [
        Path(__file__).resolve(),
        (
            repository_root
            / "hyskill"
            / "k2_answer_provenance.py"
        ).resolve(),
        (
            repository_root
            / "hyskill"
            / "downstream_reuse.py"
        ).resolve(),
    ]
    builder_code_files: list[CodeFileDigest] = code_file_digests(
        code_paths,
        repository_root,
    )
    builder_code_bundle_sha256: str = (
        code_bundle_sha256_from_digests(builder_code_files)
    )
    payloads: dict[str, JsonObject] = build_fleet_payloads(
        formal_root,
        evidence_root,
        raw_sources,
        qwen_run_logs,
        cast(list[JsonObject], builder_code_files),
        builder_code_bundle_sha256,
        PRODUCTION_CONTRACT,
    )
    output_hashes: dict[str, str] = write_payloads_new_atomic(
        evidence_root,
        payloads,
    )
    print(
        canonical_json(
            {
                "event": "k2_answer_provenance_built",
                "evidence_root": str(evidence_root),
                "manifests": output_hashes,
                "rows": 56600,
                "early_raw_k2": 16980,
                "posthoc_structural": 16873,
                "formal_retry_after_import": 107,
                "accepted_public_pack": False,
            }
        )
    )


if __name__ == "__main__":
    main()
