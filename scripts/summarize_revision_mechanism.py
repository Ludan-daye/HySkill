#!/usr/bin/env python3
"""E5: Query vs. pure imagination cosine similarity against gold skills.

For each held-out instance across 7 models × 5 domains, computes:
  - cos(query, gold)
  - cos(normalize(mean(imagination_t[:2])), gold) for t in {skill, passage, sentence}
    without query anchoring and without field splitting
  - routed_template similarity using the template selected by the validation router
Multiple gold skills take the maximum cosine similarity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np

from hyskill.embedder import Embedder
from hyskill.k_ablation import (
    DOMAINS,
    require_list,
    require_object,
    require_string,
    sha256_file,
)
from hyskill.revision_rankings import (
    IMAGINATION_TEMPLATES,
    VARIANT_TEMPLATE,
    load_imagination_cache,
    resolve_encoder,
)
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
EXPECTED_TEST_COUNTS: dict[str, int] = {
    "theoremqa": 598,
    "logicbench": 608,
    "medcalcbench": 880,
    "champ": 179,
    "bigcodebench": 912,
}


def _unit_vector(vec: np.ndarray, context: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{context} must be a non-empty 1D vector: shape={arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"{context} has zero norm")
    return arr / norm


def template_mean_vector(doc_vecs: np.ndarray) -> np.ndarray:
    """Return L2-normalized mean of cached imagination embeddings (excluding query)."""

    arr = np.asarray(doc_vecs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"doc_vecs must be a non-empty 2D array: shape={arr.shape}")
    mean_vec = arr.mean(axis=0)
    return _unit_vector(mean_vec, "template_mean_vector")


def gold_similarity(vec: np.ndarray, gold_vecs: np.ndarray) -> float:
    """Return max cosine similarity between ``vec`` and one or more ``gold_vecs``."""

    unit_v = _unit_vector(vec, "vec")
    g_arr = np.asarray(gold_vecs, dtype=np.float64)
    if g_arr.ndim != 2 or g_arr.shape[0] == 0 or g_arr.shape[1] != unit_v.shape[0]:
        raise ValueError(
            f"gold_vecs must have shape (m, {unit_v.shape[0]}): actual={g_arr.shape}"
        )
    norms = np.linalg.norm(g_arr, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("gold_vecs contains a zero-norm vector")
    unit_gold = g_arr / norms
    return float(np.max(unit_gold @ unit_v))


def _load_or_build_corpus_embeddings(
    corpus_path: Path,
    encoder_id: str,
    embedder: Embedder,
    emb_cache_dir: Path,
) -> dict[str, np.ndarray]:
    from sragents.corpus import load_corpus, skill_text

    corpus = load_corpus(corpus_path)
    corpus_ids = [require_string(s.get("skill_id"), "corpus.skill_id") for s in corpus]
    corpus_texts = [skill_text(s) for s in corpus]
    sample = "".join(corpus_texts[:100])[:20000]
    raw_key = "\n".join(corpus_ids) + "|" + sample + "|" + encoder_id
    key = hashlib.sha256(raw_key.encode()).hexdigest()
    cache_file = emb_cache_dir / f"naive_hyde-{key}.npz"
    if cache_file.is_file():
        matrix = np.load(cache_file, allow_pickle=False)["full"]
    else:
        emb_cache_dir.mkdir(parents=True, exist_ok=True)
        matrix = embedder.encode(corpus_texts)
        np.savez_compressed(cache_file, full=matrix)
    if matrix.shape[0] != len(corpus_ids):
        raise RuntimeError("cached corpus matrix row count mismatch")
    return {sid: matrix[idx] for idx, sid in enumerate(corpus_ids)}


def _load_domain_test_queries(instances_path: Path) -> tuple[frozenset[str], list[dict[str, object]]]:
    from sragents.cli.retrieve import _build_query

    raw_list = require_list(json.loads(instances_path.read_text(encoding="utf-8")), str(instances_path))
    annotated: list[dict[str, object]] = []
    for idx, item in enumerate(raw_list):
        inst = require_object(item, f"{instances_path}[{idx}]")
        gold_raw = inst.get("skill_annotations", [])
        if not gold_raw:
            continue
        iid = require_string(inst.get("instance_id"), "instance_id")
        gold_ids = [require_string(g, "gold") for g in require_list(gold_raw, "skill_annotations")]
        query = _build_query(cast(dict, inst))
        annotated.append({"instance_id": iid, "query": query, "gold_skill_ids": gold_ids})
    all_ids = frozenset(str(x["instance_id"]) for x in annotated)
    held_out = test_ids(all_ids)
    test_items = [x for x in annotated if str(x["instance_id"]) in held_out]
    return held_out, test_items


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
    parser = argparse.ArgumentParser(description="Summarize E5 query vs imagination gold similarity.")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--instances-dir", type=Path, required=True)
    parser.add_argument("--community-root", type=Path, default=Path("community-results"))
    parser.add_argument("--emb-cache-dir", type=Path, default=Path("results/revision-ijcai27/emb-cache"))
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
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    corpus_path = args.corpus.resolve()
    instances_dir = args.instances_dir.resolve()
    community_root = args.community_root.resolve()
    emb_cache_dir = args.emb_cache_dir.resolve()
    out_dir = args.output_dir.resolve()
    k = int(args.k)
    samples = int(args.bootstrap_samples)
    seed = int(args.bootstrap_seed)

    snapshot_dir, encoder_rev = resolve_encoder(args.encoder, None)
    embedder = Embedder(model=None, model_name=snapshot_dir)
    corpus_vecs = _load_or_build_corpus_embeddings(corpus_path, snapshot_dir, embedder, emb_cache_dir)

    input_hashes: dict[str, str] = {str(corpus_path): sha256_file(corpus_path)}
    domain_test_ids: dict[str, frozenset[str]] = {}
    domain_items: dict[str, list[dict[str, object]]] = {}
    unique_queries: list[str] = []
    seen_q: set[str] = set()

    for domain in DOMAINS:
        inst_path = instances_dir / f"{domain}.json"
        input_hashes[str(inst_path)] = sha256_file(inst_path)
        held_out, items = _load_domain_test_queries(inst_path)
        if len(held_out) != EXPECTED_TEST_COUNTS[domain]:
            raise ValueError(f"test count mismatch in E5: {domain}={len(held_out)}")
        domain_test_ids[domain] = held_out
        domain_items[domain] = items
        for item in items:
            q = str(item["query"])
            if q not in seen_q:
                seen_q.add(q)
                unique_queries.append(q)

    q_matrix = embedder.encode(unique_queries)
    q_vec_map = {q: q_matrix[i] for i, q in enumerate(unique_queries)}

    query_rows: list[dict[str, object]] = []
    template_rows: dict[str, list[dict[str, object]]] = {
        t: [] for t in (*IMAGINATION_TEMPLATES, "routed_template")
    }

    for model in MODELS:
        cache_path = community_root / model / "imagination_full_k2.jsonl.gz"
        router_path = community_root / model / "k2" / "router_decisions.json"
        input_hashes[str(cache_path)] = sha256_file(cache_path)
        input_hashes[str(router_path)] = sha256_file(router_path)

        cache = load_imagination_cache(cache_path)
        router_payload = require_object(json.loads(router_path.read_text(encoding="utf-8")), str(router_path))
        router_domains = require_object(router_payload.get("domains"), "router.domains")

        # Batch-encode all cached imagination strings needed for test items of this model
        texts_to_encode: list[str] = []
        text_index: dict[str, int] = {}
        for domain in DOMAINS:
            for item in domain_items[domain]:
                q = str(item["query"])
                for t in IMAGINATION_TEMPLATES:
                    docs = cache[q][t][:k]
                    if len(docs) < k:
                        raise ValueError(f"fewer than {k} docs for {model}:{domain}:{item['instance_id']}:{t}")
                    for d_text in docs:
                        if d_text not in text_index:
                            text_index[d_text] = len(texts_to_encode)
                            texts_to_encode.append(d_text)
        doc_matrix = embedder.encode(texts_to_encode)

        for domain in DOMAINS:
            dom_obj = require_object(router_domains.get(domain), f"router.{domain}")
            pick_variant = require_string(dom_obj.get("pick"), f"router.{domain}.pick")
            routed_tpl = VARIANT_TEMPLATE[pick_variant]
            for item in domain_items[domain]:
                iid = str(item["instance_id"])
                q = str(item["query"])
                gold_ids = cast(list[str], item["gold_skill_ids"])
                g_vecs = np.vstack([corpus_vecs[gid] for gid in gold_ids])
                q_sim = gold_similarity(q_vec_map[q], g_vecs)
                query_rows.append(
                    {"model": model, "domain": domain, "instance_id": iid, "cosine": q_sim}
                )
                tpl_sims: dict[str, float] = {}
                for t in IMAGINATION_TEMPLATES:
                    docs = cache[q][t][:k]
                    d_vecs = np.vstack([doc_matrix[text_index[d_text]] for d_text in docs])
                    mean_v = template_mean_vector(d_vecs)
                    sim = gold_similarity(mean_v, g_vecs)
                    tpl_sims[t] = sim
                    template_rows[t].append(
                        {"model": model, "domain": domain, "instance_id": iid, "cosine": sim}
                    )
                template_rows["routed_template"].append(
                    {
                        "model": model,
                        "domain": domain,
                        "instance_id": iid,
                        "cosine": tpl_sims[routed_tpl],
                    }
                )

    def allow_all(_m: str, domain: str) -> frozenset[str]:
        return domain_test_ids.get(domain, frozenset())

    overall: dict[str, object] = {}
    for t_name, t_rows in template_rows.items():
        overall[t_name] = hierarchical_paired_bootstrap(
            paired_cells(t_rows, query_rows, "cosine", allow_all),
            samples=samples,
            seed=seed,
            name=f"e5:overall:{t_name}_vs_query",
        )

    by_domain: dict[str, object] = {}
    for domain in DOMAINS:
        def allow_dom(_m: str, d: str, target: str = domain) -> frozenset[str]:
            return domain_test_ids[target] if d == target else frozenset()

        dom_res: dict[str, object] = {}
        for t_name, t_rows in template_rows.items():
            dom_res[t_name] = hierarchical_paired_bootstrap(
                paired_cells(t_rows, query_rows, "cosine", allow_dom),
                samples=samples,
                seed=seed,
                name=f"e5:{domain}:{t_name}_vs_query",
            )
        by_domain[domain] = dom_res

    payload: dict[str, object] = {
        "experiment": "E5_mechanism_gold_cosine",
        "encoder": args.encoder,
        "encoder_revision": encoder_rev,
        "k_samples": k,
        "n_models": len(MODELS),
        "n_domains": len(DOMAINS),
        "overall_vs_query": overall,
        "by_domain_vs_query": by_domain,
    }

    q_overall_mean = overall["routed_template"]["mean_b"]  # type: ignore[index]
    md_lines = [
        "# E5: Query vs. Imagination Cosine Similarity to Gold Skills",
        "",
        f"- **Support:** 7 models × 5 domains (`3,177` held-out instances/model), encoder `{args.encoder}` (`{encoder_rev[:12]}`).",
        "- **Protocol:** Each template uses the L2-normalized mean of `K=2` cached imaginations **without** mixing the raw query vector and **without** multi-view field splitting; multi-gold instances take `max` cosine over gold skills.",
        "",
        "## Overall (Macro Across 7×5 Cells)",
        "",
        f"- Raw query mean cosine to gold: **`{q_overall_mean:.4f}`**",
        "",
        "| Representation | Mean Cosine | Imagination − Query | 95% CI | p |",
        "|---|---:|---:|---|---:|",
    ]
    for t_name in ("sentence", "passage", "skill", "routed_template"):
        res = overall[t_name]  # type: ignore[index]
        md_lines.append(
            f"| `{t_name}` | {res['mean_a']:.4f} | {res['difference']:+.4f} | "
            f"[{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}] | {res['p_two_sided']:.4f} |"
        )
    md_lines.extend(
        [
            "",
            "## By Domain (Imagination − Query, 7 Models)",
            "",
            "| Domain | Query | `sentence − Q` | `passage − Q` | `skill − Q` | `routed_tpl − Q` (95% CI) |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for domain in DOMAINS:
        d_map = by_domain[domain]  # type: ignore[index]
        r_obj = d_map["routed_template"]
        md_lines.append(
            f"| `{domain}` | {r_obj['mean_b']:.4f} | "
            f"{d_map['sentence']['difference']:+.4f} | "
            f"{d_map['passage']['difference']:+.4f} | "
            f"{d_map['skill']['difference']:+.4f} | "
            f"{r_obj['difference']:+.4f} [{r_obj['ci95'][0]:+.4f}, {r_obj['ci95'][1]:+.4f}] |"
        )
    md_lines.append("")

    manifest_meta: dict[str, object] = {
        "git_head": _git_head(REPO_ROOT),
        "encoder": args.encoder,
        "encoder_revision": encoder_rev,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "input_sha256": input_hashes,
    }
    _write_with_manifest(out_dir / "e5_mechanism.json", json.dumps(payload, indent=2) + "\n", manifest_meta)
    _write_with_manifest(out_dir / "e5_mechanism.md", "\n".join(md_lines), manifest_meta)


if __name__ == "__main__":
    main()
