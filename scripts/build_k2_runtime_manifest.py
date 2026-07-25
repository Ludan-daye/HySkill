#!/usr/bin/env python3
"""Build one explicit, credential-free K=2 model/runtime manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

from hyskill.downstream_reuse import (
    CodeFileDigest,
    JsonObject,
    RuntimeManifest,
    canonical_json,
    code_bundle_sha256_from_digests,
    code_file_digests,
    sha256_file,
    validate_runtime_manifest,
)


def parse_args() -> argparse.Namespace:
    """Parse immutable runtime identity and bundle inputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--chat-template-revision", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument(
        "--selector-code-files",
        required=True,
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--answer-code-files",
        required=True,
        nargs="+",
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_nonempty(value: str, field_name: str) -> str:
    """Return one non-empty explicit identity field."""

    normalized: str = value.strip()
    if not normalized:
        raise ValueError(
            f"Runtime identity field must be non-empty: field={field_name}"
        )
    return normalized


def resolve_code_paths(
    paths: list[Path],
    repository_root: Path,
) -> list[Path]:
    """Resolve code inputs against the explicit repository root."""

    output: list[Path] = []
    for path in paths:
        resolved_path: Path = (
            path.resolve()
            if path.is_absolute()
            else (repository_root / path).resolve()
        )
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Runtime bundle source file does not exist: path={resolved_path}"
            )
        output.append(resolved_path)
    return output


def write_json_atomic(path: Path, payload: RuntimeManifest) -> None:
    """Atomically write one runtime manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    """Hash data and code, validate identity, and write the manifest."""

    args = parse_args()
    repository_root: Path = cast(Path, args.repository_root).resolve()
    instances_path: Path = cast(Path, args.instances).resolve()
    corpus_path: Path = cast(Path, args.corpus).resolve()
    output_path: Path = cast(Path, args.output).resolve()
    if not repository_root.is_dir():
        raise NotADirectoryError(
            f"Repository root does not exist: path={repository_root}"
        )
    if not instances_path.is_file():
        raise FileNotFoundError(
            f"Instances file does not exist: path={instances_path}"
        )
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Corpus file does not exist: path={corpus_path}"
        )
    context_length: int = int(args.context_length)
    if context_length <= 0:
        raise ValueError(
            f"context-length must be positive: value={context_length}"
        )
    selector_paths: list[Path] = resolve_code_paths(
        cast(list[Path], args.selector_code_files),
        repository_root,
    )
    answer_paths: list[Path] = resolve_code_paths(
        cast(list[Path], args.answer_code_files),
        repository_root,
    )
    selector_code_files: list[CodeFileDigest] = code_file_digests(
        selector_paths,
        repository_root,
    )
    answer_code_files: list[CodeFileDigest] = code_file_digests(
        answer_paths,
        repository_root,
    )
    instances_sha256: str = sha256_file(instances_path)
    corpus_sha256: str = sha256_file(corpus_path)
    runtime_identity: JsonObject = {
        "model": require_nonempty(str(args.model), "model"),
        "checkpoint": require_nonempty(str(args.checkpoint), "checkpoint"),
        "tokenizer_revision": require_nonempty(
            str(args.tokenizer_revision),
            "tokenizer_revision",
        ),
        "chat_template_revision": require_nonempty(
            str(args.chat_template_revision),
            "chat_template_revision",
        ),
        "served_model": require_nonempty(
            str(args.served_model),
            "served_model",
        ),
        "vllm_version": require_nonempty(
            str(args.vllm_version),
            "vllm_version",
        ),
        "dtype": require_nonempty(str(args.dtype), "dtype"),
        "context_length": context_length,
    }
    manifest: RuntimeManifest = {
        "schema_version": "k2-runtime-manifest-v2",
        "instances_sha256": instances_sha256,
        "corpus_sha256": corpus_sha256,
        "runtime_identity": runtime_identity,
        "answer_code_bundle_sha256": code_bundle_sha256_from_digests(
            answer_code_files
        ),
        "selector_code_bundle_sha256": code_bundle_sha256_from_digests(
            selector_code_files
        ),
        "answer_code_files": answer_code_files,
        "selector_code_files": selector_code_files,
    }
    validated: RuntimeManifest = validate_runtime_manifest(
        cast(JsonObject, manifest),
        instances_sha256,
        corpus_sha256,
    )
    write_json_atomic(output_path, validated)
    summary: JsonObject = {
        "event": "k2_runtime_manifest_built",
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "model": runtime_identity["model"],
        "served_model": runtime_identity["served_model"],
        "instances_sha256": instances_sha256,
        "corpus_sha256": corpus_sha256,
        "selector_code_files": len(selector_code_files),
        "answer_code_files": len(answer_code_files),
        "selector_code_bundle_sha256": validated[
            "selector_code_bundle_sha256"
        ],
        "answer_code_bundle_sha256": validated["answer_code_bundle_sha256"],
    }
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
