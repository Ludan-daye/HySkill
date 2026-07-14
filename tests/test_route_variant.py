import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "route_variant", Path(__file__).parent.parent / "scripts" / "route_variant.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)


def _mk(tmp_path, name, gold_rank):
    """Retrieval file with 20 instances; gold sits at gold_rank (1-based),
    or nowhere if gold_rank is None."""
    results = []
    for n in range(20):
        ranked = [f"junk{n}_{r}" for r in range(10)]
        if gold_rank is not None:
            ranked[gold_rank - 1] = f"gold{n}"
        results.append({"instance_id": f"i{n:02d}", "gold_skill_ids": [f"gold{n}"],
                        "retrieved": [{"skill_id": s, "score": 1.0} for s in ranked]})
    p = tmp_path / f"ds-{name}.json"
    p.write_text(json.dumps({"metadata": {}, "results": results}))
    return str(p)


def test_picks_better_variant(tmp_path):
    files = {"naive_skill": _mk(tmp_path, "naive_skill", 5),
             "hyskill": _mk(tmp_path, "hyskill", 1)}
    routed, d = rv.route(files)
    assert d["pick"] == "hyskill"
    assert d["val_ndcg"]["hyskill"] > d["val_ndcg"]["naive_skill"] > 0
    assert routed["metadata"]["router"]["pick"] == "hyskill"
    assert not d["degenerate"]


def test_degenerate_falls_back_to_skill(tmp_path):
    files = {"naive_skill": _mk(tmp_path, "naive_skill", None),
             "hyskill": _mk(tmp_path, "hyskill", None)}
    _, d = rv.route(files)
    assert d["degenerate"] and d["pick"] == "naive_skill"


def test_split_matches_gate_rule(tmp_path):
    # 20 ids, val_frac 0.2 -> exactly 4 val ids, sampled with seed 0 from
    # the sorted id list — same rule as gate.py calibrate.
    files = {"naive_skill": _mk(tmp_path, "naive_skill", 1)}
    _, d = rv.route(files)
    assert d["n_val"] == 4 and d["seed"] == 0
