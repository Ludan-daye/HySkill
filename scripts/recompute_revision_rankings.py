#!/usr/bin/env python3
"""Recompute five fixed variants and the routed arm from an imagination cache.

The command writes per-instance metric rows and full rankings. It does not
write an aggregate ``all`` split. Each output gets a sibling manifest named
``<path>.manifest.json`` with input SHA-256 values, the encoder revision,
and git HEAD.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path

from hyskill.k_ablation import DOMAINS, FIXED_VARIANTS, sha256_file, validation_ids
from hyskill.revision_rankings import (
    RankingRow,
    instance_metrics,
    recompute_rankings,
    resolve_encoder,
    route,
)


REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse the required one-model, one-encoder recompute arguments."""

    parser = argparse.ArgumentParser(
        description="Recompute variant rankings from a committed imagination cache."
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--instances-dir", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--emb-cache-dir", required=True, type=Path)
    parser.add_argument("--out-metrics", required=True, type=Path)
    parser.add_argument("--out-rankings", required=True, type=Path)
    return parser.parse_args()


def git_head(repository_root: Path) -> str:
    """Return the current commit. A missing git command is an error."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "failed to resolve git HEAD: "
            f"cwd={repository_root}, stderr={error.stderr.strip()}"
        ) from error
    head: str = result.stdout.strip()
    if not head:
        raise RuntimeError(f"git rev-parse HEAD returned an empty commit: cwd={repository_root}")
    return head


def write_jsonl_gz(path: Path, rows: list[dict[str, object]]) -> None:
    """Atomically write deterministic gzip JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as archive:
            for row in rows:
                encoded: bytes = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
                archive.write(encoded)
    temporary.replace(path)


def write_manifest(output_path: Path, manifest: dict[str, object]) -> None:
    """Write ``<output_path>.manifest.json`` next to an output artifact."""

    manifest_path: Path = Path(str(output_path) + ".manifest.json")
    temporary: Path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def input_sha256(corpus: Path, cache: Path, instances_dir: Path) -> dict[str, str]:
    """Hash the corpus, imagination cache, encoder manifest, and instance files."""

    paths: list[Path] = [
        corpus,
        cache,
        REPO_ROOT / "results" / "revision-ijcai27" / "encoders.json",
    ]
    for domain in DOMAINS:
        paths.append(instances_dir / f"{domain}.json")
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required input does not exist: path={path}")
        hashes[str(path.resolve())] = sha256_file(path)
    return hashes


def metric_row(row: RankingRow, split: str, encoder: str, k: int) -> dict[str, object]:
    """Flatten one ranking into a per-instance metric record."""

    payload: dict[str, object] = {
        "model": row["model"],
        "domain": row["domain"],
        "arm": row["arm"],
        "instance_id": row["instance_id"],
        "split": split,
        "encoder": encoder,
        "k_samples": k,
    }
    payload.update(instance_metrics(row))
    return payload


def _require_jsonl_gz(path: Path) -> None:
    if not str(path).endswith(".jsonl.gz"):
        raise ValueError(f"output must be a jsonl.gz path: path={path}")


def main() -> None:
    """Rank every domain, route on that domain's validation ids, and write both outputs."""

    args = parse_args()
    if args.k <= 0:
        raise ValueError(f"k must be a positive integer: k={args.k}")
    if not args.tag:
        raise ValueError("tag must be a non-empty string")
    _require_jsonl_gz(args.out_metrics)
    _require_jsonl_gz(args.out_rankings)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    snapshot_path, revision = resolve_encoder(args.encoder, None)
    if not Path(snapshot_path).is_dir():
        raise FileNotFoundError(
            f"encoder snapshot does not exist: encoder={args.encoder}, path={snapshot_path}"
        )
    if not args.corpus.is_file():
        raise FileNotFoundError(f"corpus does not exist: path={args.corpus}")
    if not args.instances_dir.is_dir():
        raise NotADirectoryError(
            f"instances directory does not exist: path={args.instances_dir}"
        )
    if not args.cache.is_file():
        raise FileNotFoundError(f"imagination cache does not exist: path={args.cache}")

    metric_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        by_variant: dict[str, list[RankingRow]] = {}
        for variant in FIXED_VARIANTS:
            by_variant[variant] = recompute_rankings(
                tag=args.tag,
                domain=domain,
                variant=variant,
                encoder_name=args.encoder,
                k=args.k,
                corpus_path=args.corpus,
                instances_dir=args.instances_dir,
                cache_path=args.cache,
                emb_cache_dir=args.emb_cache_dir,
                st_model=None,
            )
        id_sets: list[frozenset[str]] = [
            frozenset(row["instance_id"] for row in rows) for rows in by_variant.values()
        ]
        if any(ids != id_sets[0] for ids in id_sets):
            raise ValueError(f"fixed variants cover different instances: domain={domain}")
        validation: frozenset[str] = validation_ids(id_sets[0])
        _pick, routed = route(by_variant, validation)
        ordered: list[RankingRow] = [
            row for variant in FIXED_VARIANTS for row in by_variant[variant]
        ]
        ordered.extend(routed)
        for row in ordered:
            ranking_rows.append(dict(row))
            split: str = "validation" if row["instance_id"] in validation else "test"
            metric_rows.append(metric_row(row, split, args.encoder, args.k))

    write_jsonl_gz(args.out_metrics, metric_rows)
    write_jsonl_gz(args.out_rankings, ranking_rows)
    manifest_base: dict[str, object] = {
        "tag": args.tag,
        "encoder_name": args.encoder,
        "encoder_revision": revision,
        "k": args.k,
        "git_head": git_head(REPO_ROOT),
        "input_sha256": input_sha256(args.corpus, args.cache, args.instances_dir),
    }
    write_manifest(args.out_metrics, {**manifest_base, "artifact": "metrics"})
    write_manifest(args.out_rankings, {**manifest_base, "artifact": "rankings"})


if __name__ == "__main__":
    main()
