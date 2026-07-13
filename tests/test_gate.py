import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gate", Path(__file__).parent.parent / "scripts" / "gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_split_sentences_filters_short():
    s = gate.split_sentences("Short. This sentence is long enough to keep.\nAnother sufficiently long line here")
    assert len(s) == 2 and all(len(x) >= 15 for x in s)


def test_novelty_ratio():
    assert gate.novelty_ratio([0.9, 0.9, 0.1, 0.2]) == 0.5
    assert gate.novelty_ratio([]) == 0.0


def test_pick_tau_conservative():
    # values sorted: 0.1(w) 0.2(w) 0.3(r) 0.4(w) — precision below t:
    # t>0.1: 1/1=1.0 ok ; t>0.2: 2/2=1.0 ok ; t>0.3: 2/3=0.67 no ; t>0.4: 3/4 no
    tau = gate.pick_tau([0.1, 0.2, 0.3, 0.4], [True, True, False, True], p_min=0.9)
    assert tau == 0.2


def test_pick_tau_none_when_unachievable():
    assert gate.pick_tau([0.1, 0.2], [False, False], p_min=0.9) is None


def test_apply_blocks_and_keeps(tmp_path):
    signals = {"signals": [
        {"instance_id": "a", "top1": "s1", "S1": 0.2, "S2": 0.9, "rel_truth_wrong": True},
        {"instance_id": "b", "top1": "s2", "S1": 0.8, "S2": 0.05, "rel_truth_wrong": False},
        {"instance_id": "c", "top1": "s3", "S1": 0.8, "S2": 0.9, "rel_truth_wrong": False},
    ]}
    taus = {"tau1": 0.5, "tau2": 0.1}
    retrieval = {"metadata": {}, "results": [
        {"instance_id": i, "gold_skill_ids": [], "retrieved": [{"skill_id": f"s{n}", "score": 1.0}]}
        for n, i in enumerate(["a", "b", "c"], 1)]}
    sp, tp, rp, op = (tmp_path / n for n in ("s.json", "t.json", "r.json", "o.json"))
    sp.write_text(json.dumps(signals)); tp.write_text(json.dumps(taus))
    rp.write_text(json.dumps(retrieval))

    class A: signals = str(sp); taus = str(tp); retrieval = str(rp); out = str(op)
    gate.cmd_apply(A)
    out = json.loads(op.read_text())
    by = {r["instance_id"]: r["retrieved"] for r in out["results"]}
    assert by["a"] == []          # blocked by S1
    assert by["b"] == []          # skipped by S2
    assert len(by["c"]) == 1      # kept
    assert out["metadata"]["gate"]["blocked"] == 1
    assert out["metadata"]["gate"]["skipped"] == 1
