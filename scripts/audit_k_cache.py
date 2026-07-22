#!/usr/bin/env python
"""Audit every cache entry required by an imagination-prefix experiment."""

import argparse
import json
from pathlib import Path
from typing import TypedDict, cast

from sragents.cli.retrieve import _build_query

from hyskill.generator import (
    PASSAGE_TEMPLATE,
    SENTENCE_TEMPLATE,
    SKILL_TEMPLATE,
    hypothetical_cache_key,
)

TEMPLATES: dict[str, str] = {
    "passage": PASSAGE_TEMPLATE,
    "skill": SKILL_TEMPLATE,
    "sentence": SENTENCE_TEMPLATE,
}
ERROR_EXAMPLE_LIMIT: int = 20


class AuditConfig(TypedDict):
    model: str
    instance_paths: list[Path]
    template_names: list[str]
    temperature: float
    cache_dir: Path
    k_samples: int
    output_path: Path


class GroupAudit(TypedDict):
    template: str
    sample_index: int
    expected: int
    present: int
    missing: int
    empty: int
    unreadable: int


def resolve_path(repository_root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value.resolve()
    return (repository_root / value).resolve()


def parse_template_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("templates must contain at least one template name")
    unknown = sorted(set(names) - set(TEMPLATES))
    if unknown:
        raise ValueError(
            "unknown template names: "
            f"received={unknown}, allowed={sorted(TEMPLATES)}")
    if len(names) != len(set(names)):
        raise ValueError(f"templates must not contain duplicates: {names}")
    return names


def parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(
        description="Audit a complete K-prefix imagination cache.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--instances", nargs="+", required=True, type=Path)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    model = str(args.model).strip()
    if not model:
        raise ValueError("model must be a non-empty string")
    k_samples = int(args.k)
    if k_samples <= 0:
        raise ValueError(
            f"k must be a positive integer; received {k_samples}")
    instance_paths = [
        resolve_path(repository_root, path)
        for path in cast(list[Path], args.instances)
    ]
    missing_instance_paths = [
        str(path) for path in instance_paths if not path.is_file()
    ]
    if missing_instance_paths:
        raise FileNotFoundError(
            "instance files do not exist: "
            f"paths={missing_instance_paths}")
    cache_dir = resolve_path(repository_root, cast(Path, args.cache_dir))
    if not cache_dir.is_dir():
        raise NotADirectoryError(f"cache directory does not exist: {cache_dir}")
    return {
        "model": model,
        "instance_paths": instance_paths,
        "template_names": parse_template_names(str(args.templates)),
        "temperature": float(args.temperature),
        "cache_dir": cache_dir,
        "k_samples": k_samples,
        "output_path": resolve_path(
            repository_root, cast(Path, args.output)),
    }


def load_queries(instance_paths: list[Path]) -> tuple[list[str], dict[str, int]]:
    queries: list[str] = []
    row_counts: dict[str, int] = {}
    for path in instance_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise TypeError(f"instance file must contain a JSON list: path={path}")
        row_counts[str(path)] = len(raw)
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise TypeError(
                    "instance must be a JSON object: "
                    f"path={path}, index={index}")
            instance = cast(dict[str, object], item)
            if instance.get("skill_annotations"):
                query = _build_query(instance)
                if not isinstance(query, str) or not query:
                    raise ValueError(
                        "SR-Agents produced an empty query: "
                        f"path={path}, index={index}")
                queries.append(query)
    unique_queries = sorted(set(queries))
    if not unique_queries:
        raise ValueError(
            f"no annotated queries found in instance files: {instance_paths}")
    return unique_queries, row_counts


def audit_group(config: AuditConfig, queries: list[str], template_name: str,
                sample_index: int) -> tuple[GroupAudit, list[dict[str, object]]]:
    template = TEMPLATES[template_name]
    model_tag = f"{config['model']}|{template[:20]}"
    present = 0
    empty = 0
    unreadable = 0
    errors: list[dict[str, object]] = []
    for query in queries:
        key = hypothetical_cache_key(
            model_tag=model_tag,
            temperature=config["temperature"],
            template=template,
            query=query,
            sample_index=sample_index,
        )
        path = config["cache_dir"] / f"{key}.txt"
        if not path.is_file():
            if len(errors) < ERROR_EXAMPLE_LIMIT:
                errors.append({
                    "kind": "missing",
                    "template": template_name,
                    "sample_index": sample_index,
                    "cache_path": str(path),
                })
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            unreadable += 1
            if len(errors) < ERROR_EXAMPLE_LIMIT:
                errors.append({
                    "kind": "unreadable",
                    "template": template_name,
                    "sample_index": sample_index,
                    "cache_path": str(path),
                    "error": str(error),
                })
            continue
        if not text.strip():
            empty += 1
            if len(errors) < ERROR_EXAMPLE_LIMIT:
                errors.append({
                    "kind": "empty",
                    "template": template_name,
                    "sample_index": sample_index,
                    "cache_path": str(path),
                })
            continue
        present += 1
    expected = len(queries)
    missing = expected - present - empty - unreadable
    return {
        "template": template_name,
        "sample_index": sample_index,
        "expected": expected,
        "present": present,
        "missing": missing,
        "empty": empty,
        "unreadable": unreadable,
    }, errors


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    config = parse_args()
    queries, row_counts = load_queries(config["instance_paths"])
    groups: list[GroupAudit] = []
    error_examples: list[dict[str, object]] = []
    for template_name in config["template_names"]:
        for sample_index in range(config["k_samples"]):
            group, errors = audit_group(
                config, queries, template_name, sample_index)
            groups.append(group)
            remaining = ERROR_EXAMPLE_LIMIT - len(error_examples)
            if remaining > 0:
                error_examples.extend(errors[:remaining])

    expected = sum(group["expected"] for group in groups)
    present = sum(group["present"] for group in groups)
    missing = sum(group["missing"] for group in groups)
    empty = sum(group["empty"] for group in groups)
    unreadable = sum(group["unreadable"] for group in groups)
    complete = missing == 0 and empty == 0 and unreadable == 0
    report: dict[str, object] = {
        "schema_version": 1,
        "model": config["model"],
        "k_samples": config["k_samples"],
        "temperature": config["temperature"],
        "templates": config["template_names"],
        "instance_rows": row_counts,
        "unique_queries": len(queries),
        "cache_dir": str(config["cache_dir"]),
        "expected": expected,
        "present": present,
        "missing": missing,
        "empty": empty,
        "unreadable": unreadable,
        "complete": complete,
        "groups": groups,
        "error_examples": error_examples,
    }
    write_report(config["output_path"], report)
    print(json.dumps({
        "output": str(config["output_path"]),
        "model": config["model"],
        "k_samples": config["k_samples"],
        "expected": expected,
        "present": present,
        "missing": missing,
        "empty": empty,
        "unreadable": unreadable,
        "complete": complete,
    }, ensure_ascii=False))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
