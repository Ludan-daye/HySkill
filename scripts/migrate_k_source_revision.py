#!/usr/bin/env python3
"""Atomically correct a K-ablation source revision and record file hashes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast


K_DIRECTORY_PATTERN: re.Pattern[str] = re.compile(r"k(?:1|2|4|8|10)")


class MigrationRecord(TypedDict):
    """Hash evidence for one migrated or already-current result file."""

    path: str
    status: str
    before_sha256: str
    after_sha256: str


def parse_args() -> argparse.Namespace:
    """Parse explicit migration boundaries and revision identities."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--old-revision", required=True)
    parser.add_argument("--new-revision", required=True)
    parser.add_argument("--expected-files", required=True, type=int)
    parser.add_argument("--audit-output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    """Load a JSON object or raise with the offending path."""

    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: path={path}")
    return cast(dict[str, object], value)


def source_revision(value: dict[str, object], path: Path) -> str:
    """Return the stamped source revision from one finalized result."""

    metadata_value: object = value.get("metadata")
    if not isinstance(metadata_value, dict):
        raise TypeError(f"Result metadata must be an object: path={path}")
    metadata: dict[str, object] = cast(dict[str, object], metadata_value)
    stamp_value: object = metadata.get("k_ablation")
    if not isinstance(stamp_value, dict):
        raise TypeError(f"K-ablation stamp must be an object: path={path}")
    stamp: dict[str, object] = cast(dict[str, object], stamp_value)
    revision_value: object = stamp.get("source_revision")
    if not isinstance(revision_value, str) or not revision_value:
        raise TypeError(f"Source revision must be a non-empty string: path={path}")
    return revision_value


def corrected_value(
    value: dict[str, object], path: Path, old_revision: str, new_revision: str
) -> tuple[dict[str, object], str]:
    """Return a corrected copy and an explicit migration status."""

    actual_revision: str = source_revision(value, path)
    if actual_revision not in (old_revision, new_revision):
        raise ValueError(
            f"Unexpected source revision: path={path}, expected_old={old_revision!r}, "
            f"expected_new={new_revision!r}, actual={actual_revision!r}"
        )
    if actual_revision == new_revision:
        return copy.deepcopy(value), "already_current"
    corrected: dict[str, object] = copy.deepcopy(value)
    metadata = cast(dict[str, object], corrected["metadata"])
    stamp = cast(dict[str, object], metadata["k_ablation"])
    stamp["source_revision"] = new_revision
    return corrected, "migrated"


def write_json_atomic(path: Path, value: object) -> None:
    """Atomically replace one compact JSON file and fsync its contents."""

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


def fixed_result_paths(results_dir: Path, expected_files: int) -> list[Path]:
    """Return the exact fixed-result inventory under canonical K directories."""

    paths: list[Path] = sorted(
        path
        for directory in results_dir.iterdir()
        if directory.is_dir() and K_DIRECTORY_PATTERN.fullmatch(directory.name)
        for path in directory.glob("*.json")
        if path.is_file()
    )
    if len(paths) != expected_files:
        raise ValueError(
            f"Fixed-result inventory mismatch: results_dir={results_dir}, "
            f"expected={expected_files}, actual={len(paths)}"
        )
    return paths


def migrate_file(
    path: Path, old_revision: str, new_revision: str
) -> MigrationRecord:
    """Correct one file atomically and return before-and-after evidence."""

    before_sha256: str = sha256_file(path)
    value: dict[str, object] = load_json_object(path)
    corrected, status = corrected_value(value, path, old_revision, new_revision)
    if status == "migrated":
        write_json_atomic(path, corrected)
    verified: dict[str, object] = load_json_object(path)
    actual_revision: str = source_revision(verified, path)
    if actual_revision != new_revision:
        raise ValueError(
            f"Source revision migration did not persist: path={path}, "
            f"expected={new_revision!r}, actual={actual_revision!r}"
        )
    return {
        "path": str(path),
        "status": status,
        "before_sha256": before_sha256,
        "after_sha256": sha256_file(path),
    }


def main() -> None:
    """Migrate the bounded fixed-result inventory and write an audit artifact."""

    args = parse_args()
    if args.expected_files <= 0:
        raise ValueError(
            f"Expected file count must be positive: value={args.expected_files}"
        )
    if args.old_revision == args.new_revision:
        raise ValueError("Old and new source revisions must differ")
    paths: list[Path] = fixed_result_paths(args.results_dir, args.expected_files)
    for path in paths:
        corrected_value(
            load_json_object(path), path, args.old_revision, args.new_revision
        )
    records: list[MigrationRecord] = [
        migrate_file(path, args.old_revision, args.new_revision) for path in paths
    ]
    audit: dict[str, object] = {
        "schema_version": 1,
        "artifact": "k_ablation_source_revision_migration",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(args.results_dir),
        "old_revision": args.old_revision,
        "new_revision": args.new_revision,
        "files": len(records),
        "migrated": sum(record["status"] == "migrated" for record in records),
        "already_current": sum(
            record["status"] == "already_current" for record in records
        ),
        "records": records,
    }
    write_json_atomic(args.audit_output, audit)
    print(json.dumps({key: audit[key] for key in ("files", "migrated", "already_current")}))


if __name__ == "__main__":
    main()
