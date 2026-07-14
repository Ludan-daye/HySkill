#!/usr/bin/env python
"""Phase 2 accuracy significance: paired bootstrap over per-instance correct
flags for arm pairs, on the full set and test-only (excluding the tau
calibration validation split). Reads results/phase2/<ds>-<arm>.eval.json and
<ds>-taus.json; writes results/phase2/significance.json.
"""

import json
import random
import sys

DOMAINS = ["theoremqa", "logicbench", "medcalcbench", "champ"]
ARMS = ["bare", "always", "gated", "select", "oracle"]
PAIRS = [("gated", "always"), ("gated", "select"), ("always", "bare"),
         ("gated", "bare"), ("select", "always")]
B = 10000


def load(ds: str, arm: str) -> dict[str, int]:
    d = json.load(open(f"results/phase2/{ds}-{arm}.eval.json"))
    return {r["instance_id"]: int(bool(r["correct"])) for r in d["details"]}


def boot(a: dict, b: dict, ids: list[str], rng: random.Random):
    diffs = [a[i] - b[i] for i in ids]
    n = len(diffs)
    obs = sum(diffs) / n
    cnt = sum(1 for _ in range(B)
              if sum(diffs[rng.randrange(n)] for _ in range(n)) / n <= 0)
    return obs, min(2 * min(cnt, B - cnt) / B, 1.0)  # two-sided


def main() -> None:
    out = {}
    rng = random.Random(0)
    for ds in DOMAINS:
        arms = {arm: load(ds, arm) for arm in ARMS}
        val = set(json.load(open(f"results/phase2/{ds}-taus.json"))["val_ids"])
        common = sorted(set.intersection(*[set(v) for v in arms.values()]))
        test = [i for i in common if i not in val]
        row = {"n": len(common), "n_test": len(test)}
        for a, b in PAIRS:
            d_full, p_full = boot(arms[a], arms[b], common, rng)
            d_test, p_test = boot(arms[a], arms[b], test, rng)
            row[f"{a}-vs-{b}"] = {"delta": round(d_full, 4), "p": round(p_full, 4),
                                  "delta_test": round(d_test, 4),
                                  "p_test": round(p_test, 4)}
        row["acc"] = {arm: round(sum(v.values()) / len(v), 4)
                      for arm, v in arms.items()}
        row["acc_test"] = {arm: round(sum(arms[arm][i] for i in test) / len(test), 4)
                           for arm in arms}
        out[ds] = row
        print(ds, "done", file=sys.stderr)

    # pooled across all domains (deployable arms)
    pooled = {}
    merged = {arm: {} for arm in ARMS[:-1]}
    for arm in merged:
        for ds in DOMAINS:
            merged[arm].update({f"{ds}|{k}": v for k, v in load(ds, arm).items()})
    ids = sorted(merged["bare"])
    for a, b in PAIRS:
        d, p = boot(merged[a], merged[b], ids, rng)
        pooled[f"{a}-vs-{b}"] = {"delta": round(d, 4), "p": round(p, 4)}
    out["pooled"] = pooled

    json.dump(out, open("results/phase2/significance.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
