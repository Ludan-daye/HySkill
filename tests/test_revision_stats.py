"""Tests for the IJCAI-27 paired hierarchical bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

THEOREMQA = (
    Path(__file__).parents[1] / "external" / "SR-Agents" / "data" / "bench" / "instances" / "theoremqa.json"
)


def test_split_counts() -> None:
    from hyskill.k_ablation import validation_ids
    from hyskill.revision_stats import test_ids

    payload = json.loads(THEOREMQA.read_text(encoding="utf-8"))
    domain_ids = frozenset(row["instance_id"] for row in payload)
    assert len(domain_ids) == 747
    assert len(validation_ids(domain_ids)) == 149
    assert len(test_ids(domain_ids)) == 747 - 149


def test_paired_requires_identical_ids() -> None:
    from hyskill.revision_stats import paired_cells

    rows_a = [
        {"model": "glm4-9b", "domain": "theoremqa", "instance_id": "a", "hit@10": 1.0},
    ]
    rows_b = [
        {"model": "glm4-9b", "domain": "theoremqa", "instance_id": "b", "hit@10": 1.0},
    ]

    def allow(_model: str, _domain: str) -> frozenset[str]:
        return frozenset({"a", "b"})

    with pytest.raises(ValueError, match="different instance ids"):
        paired_cells(rows_a, rows_b, "hit@10", allow)


def test_bootstrap_zero_difference() -> None:
    from hyskill.revision_stats import hierarchical_paired_bootstrap

    values = np.asarray([1.0, 0.0, 1.0, 0.5], dtype=np.float64)
    cells = {("glm4-9b", "theoremqa"): (values, values.copy())}
    result = hierarchical_paired_bootstrap(cells, samples=64, seed=0, name="zero")
    assert result["ci95"] == [0.0, 0.0]
    assert result["difference"] == 0.0


def test_bootstrap_deterministic() -> None:
    from hyskill.revision_stats import hierarchical_paired_bootstrap

    left = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    right = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    cells = {
        ("llama31-8b", "logicbench"): (left, right),
        ("glm4-9b", "theoremqa"): (right, left),
        ("llama31-8b", "theoremqa"): (left, left),
        ("glm4-9b", "logicbench"): (right, right),
    }
    first = hierarchical_paired_bootstrap(cells, samples=40, seed=7, name="depth")
    second = hierarchical_paired_bootstrap(cells, samples=40, seed=7, name="depth")
    assert first == second


def test_comparison_seed_matches_k_ablation() -> None:
    from scripts.summarize_k_ablation import comparison_seed as ablation_seed

    from hyskill.revision_stats import comparison_seed

    for base_seed, key in ((0, "gated"), (7, "e1:glm4-9b:theoremqa:instances"), (99, "")):
        assert comparison_seed(base_seed, key) == ablation_seed(base_seed, key)
