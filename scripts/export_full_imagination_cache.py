#!/usr/bin/env python
"""Export a complete imagination-prefix cache as a GitHub-ready artifact.

The raw cache contains tens of thousands of hashed files and must not be
committed. This exporter verifies every required cache entry and writes one
deterministic gzip JSONL file plus a provenance manifest.
"""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import TypedDict, cast

from sragents.cli.retrieve import _build_query

from hyskill.generator import (
    PASSAGE_TEMPLATE,
    SENTENCE_TEMPLATE,
    SKILL_TEMPLATE,
    hypothetical_cache_key,
)

DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
    "bigcodebench",
)
EXPECTED_DOMAIN_COUNTS: dict[str, int] = {
    "theoremqa": 747,
    "logicbench": 760,
    "medcalcbench": 1100,
    "champ": 223,
    "bigcodebench": 1140,
}
TEMPLATES: tuple[tuple[str, str], ...] = (
    ("sentence", SENTENCE_TEMPLATE),
    ("passage", PASSAGE_TEMPLATE),
    ("skill", SKILL_TEMPLATE),
)
TEMPERATURE: float = 0.7
EXPECTED_TOTAL_ROWS: int = 3970
EXPECTED_UNIQUE_QUERIES: int = 3968
MAX_GITHUB_ARTIFACT_BYTES: int = 95_000_000
TAG_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9._-]+$")


class ExportConfig(TypedDict):
    tag: str
    model: str
    model_revision: str
    generation_commit: str
    k_samples: int
    cache_dir: Path
    instances_dir: Path
    repository_root: Path


class ImaginationRow(TypedDict):
    instance_id: str
    domain: str
    query: str
    imaginations: dict[str, list[str]]


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(
        description="Export a complete, verified imagination-prefix cache pack.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--generation-commit", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--instances-dir", required=True, type=Path)
    args = parser.parse_args()

    tag = str(args.tag)
    if TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(
            "tag must contain only letters, digits, dots, underscores, or "
            f"hyphens; received {tag!r}")

    model = str(args.model).strip()
    model_revision = str(args.model_revision).strip()
    generation_commit = str(args.generation_commit).strip()
    for name, value in (
            ("model", model),
            ("model_revision", model_revision),
            ("generation_commit", generation_commit)):
        if not value:
            raise ValueError(f"{name} must be a non-empty string")
    k_samples = int(args.k)
    if k_samples <= 0:
        raise ValueError(
            f"k must be a positive integer; received {k_samples}")

    repository_root = Path(__file__).resolve().parents[1]
    cache_dir = resolve_path(repository_root, cast(Path, args.cache_dir))
    instances_dir = resolve_path(repository_root, cast(Path, args.instances_dir))
    if not cache_dir.is_dir():
        raise NotADirectoryError(f"cache directory does not exist: {cache_dir}")
    if not instances_dir.is_dir():
        raise NotADirectoryError(
            f"instances directory does not exist: {instances_dir}")

    return {
        "tag": tag,
        "model": model,
        "model_revision": model_revision,
        "generation_commit": generation_commit,
        "k_samples": k_samples,
        "cache_dir": cache_dir,
        "instances_dir": instances_dir,
        "repository_root": repository_root,
    }


def resolve_path(repository_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repository_root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def repository_commit(repository_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "failed to resolve the repository commit: "
            f"cwd={repository_root}, stderr={error.stderr.strip()}") from error
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError(
            f"git returned an empty commit for repository {repository_root}")
    return commit


def load_instances(path: Path, domain: str) -> list[dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(
            f"instance file must contain a JSON list: domain={domain}, path={path}")
    instances: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(
                "instance must be a JSON object: "
                f"domain={domain}, index={index}, path={path}")
        instances.append(cast(dict[str, object], item))
    expected = EXPECTED_DOMAIN_COUNTS[domain]
    if len(instances) != expected:
        raise ValueError(
            "unexpected instance count: "
            f"domain={domain}, expected={expected}, actual={len(instances)}, "
            f"path={path}")
    return instances


def instance_id(instance: dict[str, object], domain: str, index: int) -> str:
    value = instance.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "instance_id must be a non-empty string: "
            f"domain={domain}, index={index}, value={value!r}")
    return value


def read_cached_imaginations(cache_dir: Path, model: str, query: str,
                             domain: str, iid: str, k_samples: int) -> tuple[
                                 dict[str, list[str]], set[Path]]:
    imaginations: dict[str, list[str]] = {}
    verified_paths: set[Path] = set()
    for template_name, template in TEMPLATES:
        model_tag = f"{model}|{template[:20]}"
        samples: list[str] = []
        for sample_index in range(k_samples):
            key = hypothetical_cache_key(
                model_tag=model_tag,
                temperature=TEMPERATURE,
                template=template,
                query=query,
                sample_index=sample_index,
            )
            sample_path = cache_dir / f"{key}.txt"
            if not sample_path.is_file():
                raise FileNotFoundError(
                    "required imagination cache entry is missing: "
                    f"model={model}, domain={domain}, instance_id={iid}, "
                    f"template={template_name}, sample_index={sample_index}, "
                    f"path={sample_path}")
            text = sample_path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(
                    "required imagination cache entry is empty: "
                    f"model={model}, domain={domain}, instance_id={iid}, "
                    f"template={template_name}, sample_index={sample_index}, "
                    f"path={sample_path}")
            samples.append(text)
            verified_paths.add(sample_path.resolve())
        imaginations[template_name] = samples
    return imaginations, verified_paths


def build_rows(config: ExportConfig) -> tuple[
        list[ImaginationRow], dict[str, dict[str, object]], set[Path]]:
    rows: list[ImaginationRow] = []
    instance_manifest: dict[str, dict[str, object]] = {}
    verified_paths: set[Path] = set()
    for domain in DOMAINS:
        path = config["instances_dir"] / f"{domain}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"instance file does not exist: domain={domain}, path={path}")
        instances = load_instances(path, domain)
        instance_manifest[domain] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(instances),
        }
        for index, instance in enumerate(instances):
            iid = instance_id(instance, domain, index)
            query = _build_query(instance)
            if not isinstance(query, str) or not query:
                raise ValueError(
                    "SR-Agents produced an empty query: "
                    f"domain={domain}, instance_id={iid}")
            imaginations, sample_paths = read_cached_imaginations(
                cache_dir=config["cache_dir"],
                model=config["model"],
                query=query,
                domain=domain,
                iid=iid,
                k_samples=config["k_samples"],
            )
            rows.append({
                "instance_id": iid,
                "domain": domain,
                "query": query,
                "imaginations": imaginations,
            })
            verified_paths.update(sample_paths)
    return rows, instance_manifest, verified_paths


def write_deterministic_jsonl_gzip(path: Path,
                                   rows: list[ImaginationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("wb") as destination:
        with gzip.GzipFile(
                filename="", mode="wb", fileobj=destination, mtime=0) as archive:
            for row in rows:
                encoded = (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
                archive.write(encoded)
    temporary_path.replace(path)


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def verify_written_artifact(path: Path) -> int:
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "exported artifact contains invalid JSON: "
                    f"path={path}, line={line_number}, error={error}") from error
            if not isinstance(row, dict):
                raise TypeError(
                    "exported artifact row must be a JSON object: "
                    f"path={path}, line={line_number}")
            rows += 1
    if rows != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            "exported artifact row count mismatch: "
            f"path={path}, expected={EXPECTED_TOTAL_ROWS}, actual={rows}")
    return rows


def main() -> None:
    config = parse_args()
    rows, instance_manifest, verified_paths = build_rows(config)
    unique_queries = len({row["query"] for row in rows})
    k_samples = config["k_samples"]
    expected_cache_files = EXPECTED_UNIQUE_QUERIES * len(TEMPLATES) * k_samples
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"unexpected total rows: expected={EXPECTED_TOTAL_ROWS}, "
            f"actual={len(rows)}")
    if unique_queries != EXPECTED_UNIQUE_QUERIES:
        raise ValueError(
            f"unexpected unique queries: expected={EXPECTED_UNIQUE_QUERIES}, "
            f"actual={unique_queries}")
    if len(verified_paths) != expected_cache_files:
        raise ValueError(
            "unexpected verified cache file count: "
            f"expected={expected_cache_files}, actual={len(verified_paths)}")

    output_dir = config["repository_root"] / "community-results" / config["tag"]
    artifact_path = output_dir / f"imagination_full_k{k_samples}.jsonl.gz"
    manifest_path = output_dir / f"imagination_full_k{k_samples}.manifest.json"
    write_deterministic_jsonl_gzip(artifact_path, rows)
    verified_output_rows = verify_written_artifact(artifact_path)
    artifact_size = artifact_path.stat().st_size
    if artifact_size > MAX_GITHUB_ARTIFACT_BYTES:
        raise ValueError(
            "compressed artifact exceeds the safe GitHub size limit: "
            f"path={artifact_path}, bytes={artifact_size}, "
            f"limit={MAX_GITHUB_ARTIFACT_BYTES}")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": "complete_imagination_prefix_cache",
        "tag": config["tag"],
        "model": config["model"],
        "model_revision": config["model_revision"],
        "export_code_commit": repository_commit(config["repository_root"]),
        "generation_code_commit": config["generation_commit"],
        "k_samples": k_samples,
        "temperature": TEMPERATURE,
        "templates": {
            name: {"sha256": sha256_text(template)}
            for name, template in TEMPLATES
        },
        "instances": instance_manifest,
        "rows": len(rows),
        "verified_output_rows": verified_output_rows,
        "unique_queries": unique_queries,
        "verified_cache_files": len(verified_paths),
        "output": {
            "path": str(artifact_path.relative_to(config["repository_root"])),
            "bytes": artifact_size,
            "sha256": sha256_file(artifact_path),
        },
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps({
        "artifact": str(artifact_path),
        "manifest": str(manifest_path),
        "rows": len(rows),
        "unique_queries": unique_queries,
        "verified_cache_files": len(verified_paths),
        "bytes": artifact_size,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
