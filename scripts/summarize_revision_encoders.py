#!/usr/bin/env python3
"""E4: Retrieval robustness across embedding encoders.

Summarizes held-out nDCG@10 and Recall@50 across all seven models and five
domains for each evaluated encoder, re-selecting router winners on each
encoder's own validation split.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from hyskill.k_ablation import (
    ALL_VARIANTS,
    DOMAINS,
    require_float,
    require_object,
    sha256_file,
)
from hyskill.revision_rankings import instance_metrics, load_committed_rankings
from hyskill.revision_stats import (
    hierarchical_paired_bootstrap,
    paired_cells,
    test_ids,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS: tuple[str, ...] = (
    "deepseek7b",
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
    "yi15-9b",
)
# k4 BM25 for qwen3.5-4b-reference came from SR-Agents BM25Retriever, so BM25
# contrasts use only the four models whose k4 BM25 is FastBM25.
FULL_BM25_MODELS: tuple[str, ...] = (
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen35-9b",
)
EXPECTED_TEST_COUNTS: dict[str, int] = {
    "theoremqa": 598,
    "logicbench": 608,
    "medcalcbench": 880,
    "champ": 179,
    "bigcodebench": 912,
}
REPORT_METRICS: tuple[str, ...] = ("nDCG@10", "Recall@50")


def validate_encoder_inputs(
    rows_by_encoder: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[str]:
    """Validate identical model/domain/arm coverage across all encoders."""

    if not rows_by_encoder:
        raise ValueError("rows_by_encoder must not be empty")
    reference_models: set[str] | None = None
    reference_cells: set[tuple[str, str, str]] | None = None
    reference_encoder: str | None = None

    for encoder_name, rows in rows_by_encoder.items():
        if not rows:
            raise ValueError(f"encoder has no rows: encoder={encoder_name}")
        models_present = {str(r.get("model", "")) for r in rows if r.get("split") == "test"}
        cells_present = {
            (str(r.get("model", "")), str(r.get("domain", "")), str(r.get("arm", "")))
            for r in rows
            if r.get("split") == "test"
        }
        if set(models_present) != set(MODELS):
            raise ValueError(
                f"encoder model set does not match the 7-model support: "
                f"encoder={encoder_name}, models={sorted(models_present)}"
            )
        expected_cells = {
            (m, d, a) for m in MODELS for d in DOMAINS for a in ALL_VARIANTS
        }
        if cells_present != expected_cells:
            missing = sorted(expected_cells - cells_present)[:5]
            raise ValueError(
                f"encoder is missing expected model-domain-arm cells: "
                f"encoder={encoder_name}, missing_sample={missing}"
            )
        if reference_models is None:
            reference_models = models_present
            reference_cells = cells_present
            reference_encoder = encoder_name
        elif models_present != reference_models or cells_present != reference_cells:
            raise ValueError(
                f"mixed model/cell sets across encoders: "
                f"{reference_encoder}={sorted(reference_models)} vs "
                f"{encoder_name}={sorted(models_present)}"
            )
    return sorted(reference_models or set())


def _read_jsonl_gz(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing metrics file: path={path}")
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                rows.append(require_object(json.loads(line), f"{path}:{line_no}"))
    return rows


def _macro_means(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "test":
            grouped[(str(row["arm"]), str(row["model"]), str(row["domain"]))].append(row)
    table: dict[str, dict[str, float]] = {}
    for arm in ALL_VARIANTS:
        table[arm] = {}
        for metric in REPORT_METRICS:
            cell_means = [
                sum(require_float(r.get(metric), metric) for r in grouped[(arm, m, d)])
                / len(grouped[(arm, m, d)])
                for m in MODELS
                for d in DOMAINS
            ]
            table[arm][metric] = sum(cell_means) / len(cell_means)
    return table


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _rel_path_str(raw: str | Path) -> str:
    p = Path(raw).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        parts = p.parts
        if "results" in parts:
            idx = parts.index("results")
            return "/".join(parts[idx:])
        return p.name


def _write_with_manifest(path: Path, content: str, manifest_meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

    sanitized_meta = dict(manifest_meta)
    if isinstance(sanitized_meta.get("input_sha256"), dict):
        sanitized_meta["input_sha256"] = {
            _rel_path_str(k): v
            for k, v in sanitized_meta["input_sha256"].items()  # type: ignore[union-attr]
        }

    m_path = Path(str(path) + ".manifest.json")
    manifest = {
        **sanitized_meta,
        "output_path": _rel_path_str(path),
        "output_sha256": sha256_file(path),
        "output_bytes": path.stat().st_size,
    }
    tmp_m = m_path.with_name(f"{m_path.name}.tmp")
    tmp_m.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_m.replace(m_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E4 multi-encoder retrieval results.")
    parser.add_argument(
        "--encoder-spec",
        action="append",
        required=True,
        metavar="NAME=SLUG",
        help="Repeatable encoder spec, e.g. BAAI/bge-base-en-v1.5=bge-base-en-v1.5",
    )
    parser.add_argument("--community-root", type=Path, default=Path("community-results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("community-results/revision-ijcai27-fleet"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    community_root = args.community_root.resolve()
    out_dir = args.output_dir.resolve()
    samples = int(args.bootstrap_samples)
    seed = int(args.bootstrap_seed)

    specs: list[tuple[str, str]] = []
    for raw_spec in args.encoder_spec:
        name, sep, slug = str(raw_spec).partition("=")
        if sep != "=" or not name.strip() or not slug.strip():
            raise ValueError(f"invalid --encoder-spec (expected NAME=SLUG): {raw_spec!r}")
        specs.append((name.strip(), slug.strip()))

    rows_by_encoder: dict[str, list[dict[str, object]]] = {}
    input_hashes: dict[str, str] = {}
    for enc_name, slug in specs:
        enc_rows: list[dict[str, object]] = []
        for model in MODELS:
            m_path = community_root / model / "revision-ijcai27" / f"metrics_{slug}.jsonl.gz"
            input_hashes[str(m_path)] = sha256_file(m_path)
            enc_rows.extend(_read_jsonl_gz(m_path))
        rows_by_encoder[enc_name] = enc_rows

    validate_encoder_inputs(rows_by_encoder)

    # Load BM25 for the 4 FastBM25 models
    bm25_rows: list[dict[str, object]] = []
    for model in FULL_BM25_MODELS:
        k4_path = community_root / model / "k4" / "retrieval_top50.jsonl.gz"
        input_hashes[str(k4_path)] = sha256_file(k4_path)
        for r in load_committed_rankings(model, "bm25", root=community_root):
            bm25_rows.append(
                {
                    "model": model,
                    "domain": r["domain"],
                    "arm": "bm25",
                    "instance_id": r["instance_id"],
                    **instance_metrics(r),
                }
            )

    # Compute held-out test IDs from the first encoder
    first_rows = rows_by_encoder[specs[0][0]]
    full_ids_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in first_rows:
        if row.get("arm") == "routed":
            full_ids_by_cell[(str(row["model"]), str(row["domain"]))].add(str(row["instance_id"]))
    allowed_7: dict[tuple[str, str], frozenset[str]] = {}
    for m in MODELS:
        for d in DOMAINS:
            held_out = test_ids(frozenset(full_ids_by_cell[(m, d)]))
            if len(held_out) != EXPECTED_TEST_COUNTS[d]:
                raise ValueError(f"test count mismatch for {m}:{d}: {len(held_out)}")
            allowed_7[(m, d)] = held_out

    def allow_seven(m: str, d: str) -> frozenset[str]:
        return allowed_7.get((m, d), frozenset())

    def allow_five_bm25(m: str, d: str) -> frozenset[str]:
        return allowed_7.get((m, d), frozenset()) if m in FULL_BM25_MODELS else frozenset()

    encoders_out: dict[str, object] = {}
    md_lines = [
        "# E4: Retrieval Robustness Across Embedding Encoders",
        "",
        "- **Variant & HyDE comparisons:** 7 models × 5 domains (`3,177` held-out instances/model).",
        "- **BM25 comparisons:** 4 FastBM25 models × 5 domains. `deepseek7b` and `yi15-9b` store `theoremqa` only in `k4`; `qwen3.5-4b-reference`'s `k4` BM25 used SR-Agents `BM25Retriever`. None of the three is mixed into BM25 contrasts.",
        "",
    ]

    for enc_name, slug in specs:
        enc_rows = rows_by_encoder[enc_name]
        table = _macro_means(enc_rows)
        routed_test = [r for r in enc_rows if r.get("split") == "test" and r.get("arm") == "routed"]
        passage_test = [
            r for r in enc_rows if r.get("split") == "test" and r.get("arm") == "naive_passage"
        ]
        routed_four = [r for r in routed_test if str(r["model"]) in FULL_BM25_MODELS]

        vs_passage = {
            metric: hierarchical_paired_bootstrap(
                paired_cells(routed_test, passage_test, metric, allow_seven),
                samples=samples,
                seed=seed,
                name=f"e4:{slug}:routed_vs_passage:{metric}",
            )
            for metric in REPORT_METRICS
        }
        vs_bm25_four = {
            metric: hierarchical_paired_bootstrap(
                paired_cells(routed_four, bm25_rows, metric, allow_five_bm25),
                samples=samples,
                seed=seed,
                name=f"e4:{slug}:routed_vs_bm25_4models:{metric}",
            )
            for metric in REPORT_METRICS
        }
        encoders_out[enc_name] = {
            "slug": slug,
            "macro_test_means_7models": table,
            "routed_vs_naive_passage_7models": vs_passage,
            "routed_vs_bm25_4models": vs_bm25_four,
        }

        md_lines.extend(
            [
                f"## Encoder: `{enc_name}`",
                "",
                "| Arm (7-Model Macro) | nDCG@10 | Recall@50 |",
                "|---|---:|---:|",
            ]
        )
        for arm in ALL_VARIANTS:
            md_lines.append(
                f"| `{arm}` | {table[arm]['nDCG@10']:.3f} | {table[arm]['Recall@50']:.3f} |"
            )
        vp_n = vs_passage["nDCG@10"]
        vp_r = vs_passage["Recall@50"]
        vb_n = vs_bm25_four["nDCG@10"]
        vb_r = vs_bm25_four["Recall@50"]
        md_lines.extend(
            [
                "",
                f"- **Routed vs. `naive_passage` (7 models):** nDCG@10 `{vp_n['difference']:+.3f}` `[{vp_n['ci95'][0]:+.3f}, {vp_n['ci95'][1]:+.3f}]` (`p={vp_n['p_two_sided']:.4f}`); Recall@50 `{vp_r['difference']:+.3f}` `[{vp_r['ci95'][0]:+.3f}, {vp_r['ci95'][1]:+.3f}]` (`p={vp_r['p_two_sided']:.4f}`).",
                f"- **Routed vs. `bm25` (4 FastBM25 models):** nDCG@10 `{vb_n['difference']:+.3f}` `[{vb_n['ci95'][0]:+.3f}, {vb_n['ci95'][1]:+.3f}]` (`p={vb_n['p_two_sided']:.4f}`); Recall@50 `{vb_r['difference']:+.3f}` `[{vb_r['ci95'][0]:+.3f}, {vb_r['ci95'][1]:+.3f}]` (`p={vb_r['p_two_sided']:.4f}`).",
                "",
            ]
        )

    payload: dict[str, object] = {
        "experiment": "E4_additional_encoders",
        "models_7": list(MODELS),
        "models_5_bm25": list(FULL_BM25_MODELS),
        "domains": list(DOMAINS),
        "encoders": encoders_out,
    }
    manifest_meta: dict[str, object] = {
        "git_head": _git_head(REPO_ROOT),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "input_sha256": input_hashes,
    }
    _write_with_manifest(out_dir / "e4_encoders.json", json.dumps(payload, indent=2) + "\n", manifest_meta)
    _write_with_manifest(out_dir / "e4_encoders.md", "\n".join(md_lines), manifest_meta)


if __name__ == "__main__":
    main()
