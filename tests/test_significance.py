import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "significance", Path(__file__).parent.parent / "scripts" / "significance.py")
sig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sig)


def _rec(gold, ranked):
    return {"instance_id": "x", "gold_skill_ids": gold,
            "retrieved": [{"skill_id": s, "score": 1.0} for s in ranked]}


def test_recall_at_k():
    r = _rec(["g1", "g2"], ["g1", "a", "b"])
    assert sig.per_instance_metric(r, "recall@1") == 0.5
    assert sig.per_instance_metric(r, "recall@3") == 0.5


def test_ndcg_perfect_rank_is_one():
    r = _rec(["g1"], ["g1", "a", "b"])
    assert abs(sig.per_instance_metric(r, "ndcg@10") - 1.0) < 1e-9


def test_ndcg_rank2_single_gold():
    r = _rec(["g1"], ["a", "g1", "b"])
    expected = (1 / math.log2(3)) / 1.0
    assert abs(sig.per_instance_metric(r, "ndcg@10") - expected) < 1e-9


def test_ndcg_zero_when_missed():
    r = _rec(["g1"], ["a", "b"])
    assert sig.per_instance_metric(r, "ndcg@10") == 0.0
