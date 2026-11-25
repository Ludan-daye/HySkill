"""Unit tests for IJCAI-27 revision summary scripts (E1–E7)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest


def _write_gzip_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ Task 4 (E1)


def test_strict_hit_ignores_filled_positions() -> None:
    from hyskill.revision_rankings import RankingRow, instance_metrics
    from scripts.summarize_revision_depth import strict_hit

    ranked = [f"s_{idx:02d}" for idx in range(50)]
    ranked[47] = "gold_skill"  # 48th position (1-indexed)
    row: RankingRow = {
        "model": "glm4-9b",
        "domain": "theoremqa",
        "arm": "always_rerank",
        "instance_id": "theoremqa_00000",
        "gold_skill_ids": ["gold_skill"],
        "ranked_ids": ranked,
        "omitted_count": 5,
    }
    assert instance_metrics(row)["hit@50"] == 1.0
    assert strict_hit(row, 50) == 0.0

    ranked_early = [f"s_{idx:02d}" for idx in range(50)]
    ranked_early[44] = "gold_skill"  # 45th position (1-indexed), within 50 - 5 = 45
    row_early: RankingRow = {**row, "ranked_ids": ranked_early}
    assert strict_hit(row_early, 50) == 1.0
    assert strict_hit(row_early, 20) == 0.0


def test_missing_omitted_count_raises() -> None:
    from hyskill.revision_rankings import RankingRow
    from scripts.summarize_revision_depth import strict_hit

    row: RankingRow = {
        "model": "glm4-9b",
        "domain": "theoremqa",
        "arm": "always_rerank",
        "instance_id": "theoremqa_00000",
        "gold_skill_ids": ["gold_skill"],
        "ranked_ids": ["gold_skill"],
        "omitted_count": None,
    }
    with pytest.raises(ValueError, match="omitted_count"):
        strict_hit(row, 10)


# ------------------------------------------------------------ Task 5 (E2/E3/E6)


def test_best_fixed_global_pooled() -> None:
    """Pooled instance mean wins over unweighted cell-mean average."""
    from hyskill.k_ablation import FIXED_VARIANTS
    from scripts.summarize_revision_router import best_fixed_global

    rows: list[dict[str, object]] = []
    # Small domain (1 test item): naive_sentence scores 1.0, naive_passage scores 0.0
    # Large domain (10 test items): naive_sentence scores 0.2, naive_passage scores 0.5
    # Cell-macro mean: naive_sentence = 0.60 > naive_passage = 0.25
    # Pooled instance mean: naive_sentence = 3.0/11 = 0.2727 < naive_passage = 5.0/11 = 0.4545
    for variant in FIXED_VARIANTS:
        small_score = (
            1.0
            if variant == "naive_sentence"
            else (0.0 if variant == "naive_passage" else 0.1)
        )
        rows.append(
            {
                "model": "glm4-9b",
                "domain": "champ",
                "arm": variant,
                "instance_id": "champ_0",
                "split": "test",
                "nDCG@10": small_score,
                "Recall@50": small_score,
            }
        )
        large_score = (
            0.2
            if variant == "naive_sentence"
            else (0.5 if variant == "naive_passage" else 0.1)
        )
        for idx in range(10):
            rows.append(
                {
                    "model": "glm4-9b",
                    "domain": "medcalcbench",
                    "arm": variant,
                    "instance_id": f"med_{idx}",
                    "split": "test",
                    "nDCG@10": large_score,
                    "Recall@50": large_score,
                }
            )
    assert best_fixed_global(rows) == "naive_passage"


def test_lodo_excludes_held_out_domain() -> None:
    from hyskill.k_ablation import DOMAINS, FIXED_VARIANTS
    from scripts.summarize_revision_router import lodo_pick

    base_rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        for variant in FIXED_VARIANTS:
            score = 0.8 if variant == "two_stage" else 0.3
            base_rows.append(
                {
                    "model": "glm4-9b",
                    "domain": domain,
                    "arm": variant,
                    "instance_id": f"{domain}_val_0",
                    "split": "validation",
                    "nDCG@10": score,
                    "Recall@50": score,
                }
            )
    pick_before = lodo_pick(base_rows, "glm4-9b", "logicbench")
    assert pick_before == "two_stage"

    perturbed_rows: list[dict[str, object]] = []
    for row in base_rows:
        updated = dict(row)
        if row["domain"] == "logicbench":
            updated["nDCG@10"] = 1000.0 if row["arm"] == "naive_sentence" else -999.0
        perturbed_rows.append(updated)
    pick_after = lodo_pick(perturbed_rows, "glm4-9b", "logicbench")
    assert pick_after == "two_stage"


# ------------------------------------------------------------------ Task 6 (E5)


def test_gold_similarity_takes_max() -> None:
    from scripts.summarize_revision_mechanism import gold_similarity

    vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    gold_vecs = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert gold_similarity(vec, gold_vecs) == pytest.approx(0.6)


def test_template_mean_excludes_query() -> None:
    from scripts.summarize_revision_mechanism import template_mean_vector

    doc_vecs = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    # Pure template mean of the two docs is [1.0, 1.0] -> normalized [1/sqrt(2), 1/sqrt(2)].
    # Mixing a query vector like [0.0, -10.0] would flip the y sign; template_mean_vector
    # takes only doc_vecs[:2] and never accepts or mixes a query vector.
    mean_vec = template_mean_vector(doc_vecs)
    expected = np.asarray([1.0, 1.0], dtype=np.float64) / np.sqrt(2.0)
    assert np.allclose(mean_vec, expected)
    assert np.linalg.norm(mean_vec) == pytest.approx(1.0)


# ------------------------------------------------------------------ Task 7 (E4)


def test_encoder_table_rejects_mixed_model_sets() -> None:
    from hyskill.k_ablation import ALL_VARIANTS, DOMAINS
    from scripts.summarize_revision_encoders import MODELS, validate_encoder_inputs

    def make_rows(models: tuple[str, ...], encoder: str) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for model in models:
            for domain in DOMAINS:
                for arm in ALL_VARIANTS:
                    out.append(
                        {
                            "model": model,
                            "domain": domain,
                            "arm": arm,
                            "instance_id": f"{domain}_0",
                            "split": "test",
                            "encoder": encoder,
                            "k_samples": 2,
                            "nDCG@10": 0.5,
                            "Recall@50": 0.8,
                        }
                    )
        return out

    full_models = tuple(MODELS)
    enc_a = make_rows(full_models, "BAAI/bge-base-en-v1.5")
    enc_b = make_rows(full_models[:-1], "thenlper/gte-base")
    with pytest.raises(ValueError, match="model"):
        validate_encoder_inputs(
            {
                "BAAI/bge-base-en-v1.5": enc_a,
                "thenlper/gte-base": enc_b,
            }
        )


# ----------------------------------------------------------------- Task 10 (E7)


def test_grid_counts_method_failure_as_incorrect() -> None:
    from scripts.summarize_revision_pool_grid import score_answer_rows

    rows = [
        {
            "model": "glm4-9b",
            "domain": "theoremqa",
            "arm": "hy_rerank",
            "instance_id": "t_0",
            "is_validation": False,
            "correct": True,
            "failure_category": "success",
        },
        {
            "model": "glm4-9b",
            "domain": "theoremqa",
            "arm": "hy_rerank",
            "instance_id": "t_1",
            "is_validation": False,
            "correct": True,  # Even if marked True, method_failure must score 0.0 and stay in N
            "failure_category": "method_failure",
        },
        {
            "model": "glm4-9b",
            "domain": "theoremqa",
            "arm": "hy_rerank",
            "instance_id": "t_val",
            "is_validation": True,
            "correct": True,
            "failure_category": "success",
        },
    ]
    scored = score_answer_rows(rows, expected_arm="hy_rerank")
    assert len(scored) == 2
    by_id = {row["instance_id"]: row["accuracy"] for row in scored}
    assert by_id == {"t_0": 1.0, "t_1": 0.0}


def test_grid_rejects_unavailable_models() -> None:
    from scripts.summarize_revision_pool_grid import score_answer_rows

    for forbidden in ("deepseek7b", "yi15-9b"):
        with pytest.raises(ValueError, match=forbidden):
            score_answer_rows(
                [
                    {
                        "model": forbidden,
                        "domain": "theoremqa",
                        "arm": "hy_rerank",
                        "instance_id": "t_0",
                        "is_validation": False,
                        "correct": False,
                        "failure_category": "success",
                    }
                ],
                expected_arm="hy_rerank",
            )


def test_runtime_bm25_shares_rerank_candidate_pool(tmp_path: Path) -> None:
    from hyskill.revision_rankings import (
        load_runtime_bm25_rankings,
        load_runtime_rerank_rankings,
    )

    ordered = [f"s_{idx:02d}" for idx in range(50)]
    ordered[3] = "theoremqa_000"
    reranked = [ordered[3], *ordered[:3], *ordered[4:]]
    decisions = tmp_path / "glm4-9b" / "decisions"
    decisions.mkdir(parents=True)
    row = {
        "instance_id": "theoremqa_00000",
        "domain": "theoremqa",
        "model": "glm4-9b",
        "arm": "always_rerank",
        "omitted_candidate_count": 7,
        "ordered_candidate_ids": ordered,
        "reranked_candidate_ids": reranked,
    }
    (decisions / "theoremqa-always_rerank.decisions.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    bm25 = load_runtime_bm25_rankings(tmp_path, "glm4-9b")
    rerank = load_runtime_rerank_rankings(tmp_path, "glm4-9b")
    assert [r["arm"] for r in bm25] == ["bm25_shared"]
    assert bm25[0]["ranked_ids"] == ordered
    assert bm25[0]["omitted_count"] is None
    assert bm25[0]["gold_skill_ids"] == rerank[0]["gold_skill_ids"]
    assert set(bm25[0]["ranked_ids"]) == set(rerank[0]["ranked_ids"])
