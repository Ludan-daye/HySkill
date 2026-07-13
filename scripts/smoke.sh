#!/usr/bin/env bash
# End-to-end Stage-1 smoke test: plugin loads, retrieval runs, schema written.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(mktemp -d)
.venv/bin/sragents --plugin hyskill.plugin retrieve \
    --retriever hyskill \
    --retriever-arg corpus_path=tests/fixtures/tiny_corpus.json \
    --retriever-arg mock_generator=1 \
    --retriever-arg encoder_name=sentence-transformers/all-MiniLM-L6-v2 \
    --corpus tests/fixtures/tiny_corpus.json \
    --instances tests/fixtures/tiny_instances.json \
    --output "$OUT/retrieval.json" --top-k 3
.venv/bin/python - "$OUT/retrieval.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["results"] and all(r["retrieved"] for r in d["results"])
print("SMOKE OK — recall metrics:", d.get("metrics"))
PY
