#!/usr/bin/env python3
"""Build one immutable job-bound runtime manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from hyskill.runtime_matched_execution import (
    JobBoundManifest,
    JsonObject,
    JsonValue,
    RuntimeManifestError,
    build_job_bound_manifest,
    canonical_json,
    load_job_bound_manifest,
    load_json_file,
    require_json_object,
    sha256_file,
    verify_job_bound_manifest_files,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    """Parse explicit inputs for one job-bound manifest."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-facts", required=True, type=Path)
    parser.add_argument("--generation", required=True, type=Path)
    parser.add_argument(
        "--artifact",
        required=True,
        action="append",
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--code-file",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_named_artifact(value: str) -> tuple[str, Path]:
    """Parse one required NAME=PATH artifact binding."""

    name, separator, raw_path = value.partition("=")
    if separator != "=" or not name.strip() or not raw_path.strip():
        raise RuntimeManifestError(
            "Artifact must use a non-empty NAME=PATH binding: "
            f"value={value!r}"
        )
    return name.strip(), Path(raw_path).resolve()


def resolve_code_path(path: Path, repository_root: Path) -> Path:
    """Resolve one code member relative to the declared repository root."""

    if path.is_absolute():
        return path.resolve()
    return (repository_root / path).resolve()


def reject_output_aliases(
    output_path: Path,
    source_paths: tuple[Path, ...],
) -> None:
    """Reject an output path that would overwrite a manifest input."""

    aliases: list[Path] = [
        path for path in source_paths if path.resolve() == output_path
    ]
    if aliases:
        raise RuntimeManifestError(
            "Runtime manifest output must not overwrite an input: "
            f"output={output_path}, aliases={aliases}"
        )


def main() -> None:
    """Build, validate, atomically write, and re-verify one manifest."""

    args = parse_args()
    repository_root: Path = cast(Path, args.repository_root).resolve()
    runtime_facts_path: Path = cast(Path, args.runtime_facts).resolve()
    generation_path: Path = cast(Path, args.generation).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    raw_artifacts: list[str] = cast(list[str], args.artifact)
    raw_code_paths: list[Path] = cast(list[Path], args.code_file)
    artifacts: list[tuple[str, Path]] = [
        parse_named_artifact(value) for value in raw_artifacts
    ]
    code_paths: list[Path] = [
        resolve_code_path(path, repository_root)
        for path in raw_code_paths
    ]
    reject_output_aliases(
        output_path,
        (
            runtime_facts_path,
            generation_path,
            *(path for _name, path in artifacts),
            *code_paths,
        ),
    )
    runtime_facts_value: JsonValue = load_json_file(
        runtime_facts_path,
        "runtime-facts",
    )
    generation_value: JsonValue = load_json_file(
        generation_path,
        "generation",
    )
    runtime_facts: JsonObject = require_json_object(
        runtime_facts_value,
        "runtime-facts",
    )
    generation: JsonObject = require_json_object(
        generation_value,
        "generation",
    )
    manifest: JobBoundManifest = build_job_bound_manifest(
        runtime_facts,
        generation,
        artifacts,
        code_paths,
        repository_root,
    )
    write_json_atomic(output_path, manifest)
    verified_manifest: JobBoundManifest = load_job_bound_manifest(output_path)
    verify_job_bound_manifest_files(verified_manifest, repository_root)
    job: JsonObject = require_json_object(
        verified_manifest["runtime_facts"].get("job"),
        "runtime_facts.job",
    )
    summary: JsonObject = {
        "event": "runtime_matched_runtime_manifest_built",
        "job_id": job["job_id"],
        "model": job["model"],
        "domain": job["domain"],
        "arm": job["arm"],
        "artifact_count": len(verified_manifest["artifacts"]),
        "code_file_count": len(verified_manifest["code_files"]),
        "code_bundle_sha256": verified_manifest["code_bundle_sha256"],
        "runtime_manifest_sha256": sha256_file(output_path),
        "output": str(output_path),
    }
    print(canonical_json(summary), flush=True)


if __name__ == "__main__":
    main()
