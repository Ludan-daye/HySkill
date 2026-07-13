# HySkill Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `hyskill` Python package (hypothetical-skill retrieval + naive-HyDE baseline) as an SR-Agents external retriever plugin, and Phase 0 experiment scripts on SRA-Bench.

**Architecture:** Standalone package `hyskill/` in this repo. SR-Agents (cloned at `external/SR-Agents`, gitignored) is installed as a dependency and consumed only through its public `Retriever` protocol and `--plugin` CLI mechanism — zero modifications to it. All LLM/embedding calls are injectable for offline unit tests.

**Tech Stack:** Python 3.10+, pytest, numpy, sentence-transformers, rank-bm25, openai (client only), SR-Agents CLI.

**Verified interface facts** (from source inspection, 2026-07-13):
- `Retriever` protocol: `build_index(corpus_ids: list[str], corpus_texts: list[str])`, `retrieve(queries: list[str], top_k: int) -> list[list[tuple[str, float]]]`, registered via `@register("name")` from `sragents.retrieve.base`.
- CLI: `sragents --plugin hyskill.plugin retrieve --retriever hyskill --retriever-arg KEY=VALUE ...`; kwargs reach the factory as **strings**.
- `corpus_texts[i]` = `"\n".join(name, description, content)`; corpus JSON entries: `{skill_id, name, description, content}`; instances: `{instance_id, dataset, question, skill_annotations, eval_data}`.
- Retrieval output schema + Recall/nDCG metrics are computed by the CLI itself.

---

### Task 1: Package scaffold + dev environment

**Files:**
- Create: `pyproject.toml`
- Create: `hyskill/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "hyskill"
version = "0.1.0"
description = "Hypothetical skill generation for agent skill retrieval (HyDE for Skills)"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "rank-bm25>=0.2.2",
]

[project.optional-dependencies]
full = ["sentence-transformers>=2.2", "openai>=1.0"]
dev = ["pytest>=7.0"]

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["hyskill"]
```

- [ ] **Step 2: Create empty `hyskill/__init__.py` (module docstring only) and `tests/__init__.py`**

```python
"""HySkill: hypothetical skill generation for agent skill retrieval."""
```

- [ ] **Step 3: Create venv and install**

Run: `cd /root/ludandaye/reaserch/skill-LLM && python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]"`
Expected: exits 0.

- [ ] **Step 4: Sanity check**

Run: `.venv/bin/python -c "import hyskill; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml hyskill/__init__.py tests/__init__.py
git commit -m "feat: scaffold hyskill package"
```

---

### Task 2: `parser.py` — SKILL.md field splitting

**Files:**
- Create: `hyskill/parser.py`
- Test: `tests/test_parser.py`

- [ ] **Step 1: Write failing tests**

```python
from hyskill.parser import parse_fields, parse_generated

MD = """# Lah Numbers

Some prose about Lah numbers.

```python
def lah(n, k):
    return 1
```

More prose."""


def test_parse_fields_splits_code_and_body():
    f = parse_fields(name="Lah Numbers", description="Counting partitions.", content=MD)
    assert f["meta"] == "Lah Numbers. Counting partitions."
    assert "def lah" in f["code"]
    assert "def lah" not in f["body"]
    assert "Some prose" in f["body"]


def test_parse_fields_missing_code():
    f = parse_fields(name="X", description="Y", content="no code here")
    assert f["code"] == ""


def test_parse_generated_frontmatter():
    md = "---\nname: pdf-to-md\ndescription: Convert PDFs\n---\n1. step one\n\n```py\nx=1\n```\n"
    f = parse_generated(md)
    assert f["meta"] == "pdf-to-md. Convert PDFs"
    assert "step one" in f["body"]
    assert "x=1" in f["code"]


def test_parse_generated_no_frontmatter_falls_back():
    f = parse_generated("Just a passage.\nSecond line.")
    assert f["meta"] == "Just a passage."
    assert "Second line" in f["body"]
```

(Note: the fenced block inside `MD` must be written with escaped backticks in the actual test file — use a raw string built by concatenation: `"```python\ndef lah(n, k):\n    return 1\n```"`.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_parser.py -q`
Expected: FAIL (`ModuleNotFoundError: hyskill.parser`)

- [ ] **Step 3: Implement `hyskill/parser.py`**

```python
"""Split SKILL.md text into meta / body / code fields."""

import re

_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _split_code(content: str) -> tuple[str, str]:
    """Return (body_without_fences, concatenated_fenced_code)."""
    code_blocks = _FENCE.findall(content or "")
    body = _FENCE.sub(" ", content or "")
    return body.strip(), "\n".join(b.strip() for b in code_blocks).strip()


def parse_fields(name: str, description: str, content: str) -> dict:
    """Fields for a corpus skill entry ({skill_id, name, description, content})."""
    body, code = _split_code(content)
    meta = ". ".join(p.strip().rstrip(".") for p in (name, description) if p and p.strip())
    if meta:
        meta += "."
    return {"meta": meta, "body": body, "code": code}


def parse_generated(md: str) -> dict:
    """Fields for a generated hypothetical SKILL.md (frontmatter optional)."""
    md = md or ""
    name, description = "", ""
    m = _FRONTMATTER.match(md)
    rest = md
    if m:
        rest = md[m.end():]
        for line in m.group(1).splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                if key == "name":
                    name = val.strip()
                elif key == "description":
                    description = val.strip()
    body, code = _split_code(rest)
    if not name:
        first = next((l.strip().lstrip("# ") for l in rest.splitlines() if l.strip()), "")
        name = first
        lines = [l for l in rest.splitlines() if l.strip()]
        body_rest, code = _split_code("\n".join(lines[1:]))
        body = body_rest
    meta = ". ".join(p.strip().rstrip(".") for p in (name, description) if p and p.strip())
    return {"meta": meta, "body": body, "code": code}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_parser.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add hyskill/parser.py tests/test_parser.py
git commit -m "feat: SKILL.md field parser (meta/body/code)"
```

---

### Task 3: `fusion.py` — reciprocal rank fusion

**Files:**
- Create: `hyskill/fusion.py`
- Test: `tests/test_fusion.py`

- [ ] **Step 1: Write failing test (hand-computable example)**

```python
from hyskill.fusion import rrf


def test_rrf_hand_example():
    # Path A ranks: s1, s2 ; Path B ranks: s2, s3
    # k=0 for easy math: s1 = 1/1 = 1.0 ; s2 = 1/2 + 1/1 = 1.5 ; s3 = 1/2 = 0.5
    out = rrf([["s1", "s2"], ["s2", "s3"]], k=0, top_k=3)
    assert [i for i, _ in out] == ["s2", "s1", "s3"]
    assert abs(dict(out)["s2"] - 1.5) < 1e-9


def test_rrf_absent_id_gets_no_contribution():
    out = rrf([["a"], ["b"]], k=0, top_k=10)
    assert dict(out)["a"] == dict(out)["b"] == 1.0


def test_rrf_top_k_truncates():
    out = rrf([["a", "b", "c"]], k=60, top_k=2)
    assert len(out) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_fusion.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `hyskill/fusion.py`**

```python
"""Reciprocal rank fusion over multiple ranked id lists."""


def rrf(rankings: list[list[str]], k: int = 60, top_k: int = 50) -> list[tuple[str, float]]:
    """Fuse rankings; ids absent from a path simply get no contribution from it.

    rank is 1-based: contribution = 1 / (k + rank).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, sid in enumerate(ranking, start=1):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top_k]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_fusion.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add hyskill/fusion.py tests/test_fusion.py
git commit -m "feat: reciprocal rank fusion"
```

---

### Task 4: `generator.py` — hypothetical skill generation with cache + fallback

**Files:**
- Create: `hyskill/generator.py`
- Test: `tests/test_generator.py`

- [ ] **Step 1: Write failing tests (fake client, tmp cache dir)**

```python
from hyskill.generator import HypotheticalGenerator, SKILL_TEMPLATE, PASSAGE_TEMPLATE


class FakeClient:
    def __init__(self, reply="---\nname: x\ndescription: y\n---\nbody"):
        self.calls = 0
        self.reply = reply

    def complete(self, prompt: str, temperature: float) -> str:
        self.calls += 1
        return self.reply


class FailingClient:
    def complete(self, prompt, temperature):
        raise RuntimeError("boom")


def test_generates_k_samples(tmp_path):
    fake = FakeClient()
    g = HypotheticalGenerator(client=fake, k_samples=3, cache_dir=tmp_path)
    docs = g.generate("task q")
    assert len(docs) == 3 and fake.calls == 3


def test_cache_hit_skips_client(tmp_path):
    fake = FakeClient()
    g = HypotheticalGenerator(client=fake, k_samples=2, cache_dir=tmp_path)
    g.generate("task q")
    g2 = HypotheticalGenerator(client=fake, k_samples=2, cache_dir=tmp_path)
    g2.generate("task q")
    assert fake.calls == 2  # second run fully cached


def test_failure_returns_empty(tmp_path):
    g = HypotheticalGenerator(client=FailingClient(), k_samples=2, cache_dir=tmp_path)
    assert g.generate("task q") == []


def test_templates_mention_task():
    assert "{q}" in SKILL_TEMPLATE and "{q}" in PASSAGE_TEMPLATE
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_generator.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `hyskill/generator.py`**

```python
"""Hypothetical document generation via an OpenAI-compatible endpoint.

The network client is injectable; the default OpenAIClient is built lazily
so unit tests never import `openai`.
"""

import hashlib
import json
import os
from pathlib import Path

SKILL_TEMPLATE = (
    "You are writing a SKILL.md that an agent would use to solve the task below. "
    "Output: (1) frontmatter with `name` and a one-line `description`; "
    "(2) numbered procedure steps; (3) a minimal code skeleton in a fenced block. "
    "Be concise, at most 300 tokens. Factual precision is not required — capture "
    "what the right skill would look like.\n\nTask: {q}"
)

PASSAGE_TEMPLATE = (
    "Write a short passage that solves or answers the task below. "
    "Factual precision is not required.\n\nTask: {q}"
)


class OpenAIClient:
    """Thin wrapper; requires `openai` and OPENAI_API_KEY (any value for local servers)."""

    def __init__(self, model: str, api_base: str):
        from openai import OpenAI
        self._client = OpenAI(base_url=api_base,
                              api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self._model = model

    def complete(self, prompt: str, temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=400,
        )
        return resp.choices[0].message.content or ""


class HypotheticalGenerator:
    def __init__(self, client, k_samples: int = 4, temperature: float = 0.7,
                 template: str = SKILL_TEMPLATE, cache_dir=None, model_tag: str = ""):
        self._client = client
        self._k = k_samples
        self._temp = temperature
        self._template = template
        self._cache = Path(cache_dir) if cache_dir else None
        self._tag = model_tag
        self.n_failures = 0

    def _key(self, query: str, i: int) -> str:
        raw = json.dumps([self._tag, self._temp, self._template, query, i])
        return hashlib.sha256(raw.encode()).hexdigest()

    def generate(self, query: str) -> list[str]:
        """Return up to k_samples hypothetical docs; [] if all attempts fail."""
        docs = []
        prompt = self._template.format(q=query)
        for i in range(self._k):
            cached = None
            if self._cache:
                p = self._cache / f"{self._key(query, i)}.txt"
                if p.exists():
                    cached = p.read_text()
            if cached is not None:
                docs.append(cached)
                continue
            text = None
            for _ in range(2):  # one retry
                try:
                    text = self._client.complete(prompt, self._temp)
                    break
                except Exception:
                    continue
            if text is None:
                self.n_failures += 1
                continue
            if self._cache:
                self._cache.mkdir(parents=True, exist_ok=True)
                (self._cache / f"{self._key(query, i)}.txt").write_text(text)
            docs.append(text)
        return docs
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_generator.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add hyskill/generator.py tests/test_generator.py
git commit -m "feat: hypothetical generator with disk cache and failure fallback"
```

---

### Task 5: `embedder.py` — injectable encoder wrapper

**Files:**
- Create: `hyskill/embedder.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: Write failing test (deterministic fake model)**

```python
import numpy as np
from hyskill.embedder import Embedder


class FakeST:
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(8)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def test_encode_shape_and_norm():
    e = Embedder(model=FakeST())
    v = e.encode(["a", "b"])
    assert v.shape == (2, 8)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_encode_empty_returns_empty():
    e = Embedder(model=FakeST())
    assert e.encode([]).shape[0] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_embedder.py -q`
Expected: FAIL

- [ ] **Step 3: Implement `hyskill/embedder.py`**

```python
"""Sentence-transformers wrapper with lazy loading and injectable model."""

import numpy as np


class Embedder:
    def __init__(self, model=None, model_name: str = "BAAI/bge-base-en-v1.5",
                 batch_size: int = 256):
        self._model = model
        self._name = model_name
        self._bs = batch_size

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        emb = np.asarray(self._load().encode(
            texts, batch_size=self._bs, show_progress_bar=len(texts) > 1000,
            normalize_embeddings=True,
        ))
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_embedder.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add hyskill/embedder.py tests/test_embedder.py
git commit -m "feat: embedder wrapper (lazy sentence-transformers, injectable)"
```

---

### Task 6: `retriever.py` — HySkillRetriever (core)

**Files:**
- Create: `hyskill/retriever.py`
- Test: `tests/test_retriever.py`
- Create: `tests/fixtures/tiny_corpus.json` (6 synthetic skills: 4 with code blocks, 2 without)

- [ ] **Step 1: Write `tests/fixtures/tiny_corpus.json`**

```json
[
  {"skill_id": "s_pdf", "name": "PDF to Markdown", "description": "Convert scanned PDFs to Markdown with tables.", "content": "# PDF to Markdown\n\nOCR pages then rebuild tables.\n\n```python\nocr(pdf)\n```"},
  {"skill_id": "s_csv", "name": "CSV Cleaner", "description": "Clean malformed CSV files.", "content": "# CSV Cleaner\n\nFix quoting and delimiters.\n\n```python\nclean(csv)\n```"},
  {"skill_id": "s_lah", "name": "Lah Numbers", "description": "Compute Lah numbers for ordered partitions.", "content": "# Lah Numbers\n\nUse L(n,k) = C(n-1,k-1) * n!/k!.\n\n```python\nlah(n, k)\n```"},
  {"skill_id": "s_web", "name": "Web Scraper", "description": "Scrape pages politely.", "content": "# Web Scraper\n\nRespect robots.txt.\n\n```python\nscrape(url)\n```"},
  {"skill_id": "s_essay", "name": "Essay Outline", "description": "Outline argumentative essays.", "content": "# Essay Outline\n\nThesis, points, conclusion. No code needed."},
  {"skill_id": "s_meet", "name": "Meeting Notes", "description": "Summarize meetings.", "content": "# Meeting Notes\n\nCapture decisions and owners."}
]
```

- [ ] **Step 2: Write failing tests**

```python
import hashlib
import json
from pathlib import Path

import numpy as np

from hyskill.retriever import HySkillRetriever

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.json"


class FakeST:
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.lower()[:40].encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(16)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


class EchoGenerator:
    """Returns a hypothetical skill that reuses the query's words, so BM25 and
    (loosely) dense paths favour lexically-overlapping corpus skills."""

    n_failures = 0

    def __init__(self, k=2):
        self._k = k

    def generate(self, query):
        md = ("---\nname: hyp\ndescription: " + query[:60] + "\n---\n"
              "1. do it\n\n```python\nsolve()\n```")
        return [md] * self._k


class EmptyGenerator:
    n_failures = 0

    def generate(self, query):
        return []


def _make(gen):
    corpus = json.loads(FIXTURE.read_text())
    r = HySkillRetriever(corpus_path=str(FIXTURE), generator=gen,
                         st_model=FakeST())
    ids = [s["skill_id"] for s in corpus]
    texts = ["\n".join([s["name"], s["description"], s["content"]]) for s in corpus]
    r.build_index(ids, texts)
    return r


def test_protocol_shapes_and_sorting():
    r = _make(EchoGenerator())
    out = r.retrieve(["convert scanned pdf to markdown tables", "compute lah numbers"], top_k=3)
    assert len(out) == 2
    for ranking in out:
        assert 0 < len(ranking) <= 3
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)
        for sid, _ in ranking:
            assert sid.startswith("s_")


def test_bm25_path_lifts_lexical_match():
    r = _make(EchoGenerator())
    out = r.retrieve(["compute Lah numbers ordered partitions"], top_k=6)
    top_ids = [sid for sid, _ in out[0][:3]]
    assert "s_lah" in top_ids


def test_fallback_on_empty_generation_still_returns():
    r = _make(EmptyGenerator())
    out = r.retrieve(["clean malformed csv file"], top_k=4)
    assert len(out[0]) == 4  # falls back to query-only retrieval


def test_codeless_skills_survivable():
    r = _make(EchoGenerator())
    out = r.retrieve(["outline an argumentative essay thesis points"], top_k=6)
    assert "s_essay" in [sid for sid, _ in out[0]]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_retriever.py -q`
Expected: FAIL (module missing)

- [ ] **Step 4: Implement `hyskill/retriever.py`**

```python
"""HySkillRetriever — hypothetical-skill multi-view retrieval (SR-Agents protocol).

Pipeline per query:
  1. generate K hypothetical SKILL.md docs (query-side HyDE);
  2. parse fields, embed, per-field centroid; query vector mixed into the
     meta centroid with weight 1/(K+1)  (HyDE eq. 8 adapted);
  3. rank per path: meta<->meta, body<->body, code<->code (code-bearing
     skills only), BM25(query -> full text);
  4. RRF-fuse paths; return top_k.
Fallback when generation fails: dense query -> full-text ranking fused with BM25.
"""

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from hyskill.embedder import Embedder
from hyskill.fusion import rrf
from hyskill.parser import parse_fields, parse_generated

_PATH_DEPTH = 100  # per-path ranking depth fed into RRF


def _tok(text: str) -> list[str]:
    return text.lower().split()


class HySkillRetriever:
    def __init__(self, corpus_path: str, generator, st_model=None,
                 encoder_name: str = "BAAI/bge-base-en-v1.5", rrf_k: int = 60):
        self._corpus_path = corpus_path
        self._generator = generator
        self._embedder = Embedder(model=st_model, model_name=encoder_name)
        self._rrf_k = rrf_k

    # ---------------------------------------------------------- indexing
    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        raw = {s["skill_id"]: s for s in json.loads(Path(self._corpus_path).read_text())}
        self._ids = list(corpus_ids)
        fields = [parse_fields(raw[i].get("name", ""), raw[i].get("description", ""),
                               raw[i].get("content", "")) if i in raw
                  else {"meta": t[:200], "body": t, "code": ""}
                  for i, t in zip(corpus_ids, corpus_texts)]

        self._meta_emb = self._embedder.encode([f["meta"] or " " for f in fields])
        self._body_emb = self._embedder.encode([f["body"] or " " for f in fields])
        self._code_ids = [i for i, f in zip(corpus_ids, fields) if f["code"]]
        self._code_emb = self._embedder.encode(
            [f["code"] for f in fields if f["code"]])
        self._full_emb = self._embedder.encode(list(corpus_texts))
        self._bm25 = BM25Okapi([_tok(t) for t in corpus_texts])

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _rank(query_vec: np.ndarray, matrix: np.ndarray, ids: list[str]) -> list[str]:
        if matrix.shape[0] == 0:
            return []
        scores = matrix @ query_vec
        order = np.argsort(-scores)[:_PATH_DEPTH]
        return [ids[j] for j in order]

    def _bm25_rank(self, query: str) -> list[str]:
        scores = self._bm25.get_scores(_tok(query))
        order = np.argsort(-scores)[:_PATH_DEPTH]
        return [self._ids[j] for j in order]

    # ---------------------------------------------------------- retrieval
    def retrieve(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        results = []
        q_vecs = self._embedder.encode(list(queries))
        for query, q_vec in zip(queries, q_vecs):
            docs = self._generator.generate(query)
            if not docs:
                paths = [self._rank(q_vec, self._full_emb, self._ids),
                         self._bm25_rank(query)]
                results.append(rrf(paths, k=self._rrf_k, top_k=top_k))
                continue
            parsed = [parse_generated(d) for d in docs]
            centroids = {}
            for field in ("meta", "body", "code"):
                texts = [p[field] for p in parsed if p[field]]
                if texts:
                    centroids[field] = self._embedder.encode(texts).mean(axis=0)
            k = len(docs)
            if "meta" in centroids:  # query anchor into meta path (HyDE eq. 8)
                centroids["meta"] = (centroids["meta"] * k + q_vec) / (k + 1)
            else:
                centroids["meta"] = q_vec
            for f in centroids:
                n = np.linalg.norm(centroids[f])
                if n > 0:
                    centroids[f] = centroids[f] / n

            paths = [self._rank(centroids["meta"], self._meta_emb, self._ids)]
            if "body" in centroids:
                paths.append(self._rank(centroids["body"], self._body_emb, self._ids))
            if "code" in centroids:
                paths.append(self._rank(centroids["code"], self._code_emb, self._code_ids))
            paths.append(self._bm25_rank(query))
            results.append(rrf(paths, k=self._rrf_k, top_k=top_k))
        return results
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_retriever.py -q`
Expected: 4 passed

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all passed

- [ ] **Step 7: Commit**

```bash
git add hyskill/retriever.py tests/test_retriever.py tests/fixtures/tiny_corpus.json
git commit -m "feat: HySkillRetriever with multi-view centroids, query anchor, 4-path RRF, fallback"
```

---

### Task 7: `naive_hyde.py` + `plugin.py` — baseline retriever and SR-Agents registration

**Files:**
- Create: `hyskill/naive_hyde.py`
- Create: `hyskill/plugin.py`
- Test: `tests/test_naive_hyde.py`, `tests/test_plugin.py`

- [ ] **Step 1: Write failing tests**

`tests/test_naive_hyde.py`:
```python
import hashlib
import json
from pathlib import Path

import numpy as np

from hyskill.naive_hyde import NaiveHydeRetriever

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.json"


class FakeST:
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.lower()[:40].encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(16)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


class OneDoc:
    n_failures = 0

    def generate(self, query):
        return ["a passage about " + query]


def test_naive_hyde_protocol():
    corpus = json.loads(FIXTURE.read_text())
    r = NaiveHydeRetriever(generator=OneDoc(), st_model=FakeST())
    ids = [s["skill_id"] for s in corpus]
    texts = ["\n".join([s["name"], s["description"], s["content"]]) for s in corpus]
    r.build_index(ids, texts)
    out = r.retrieve(["clean csv"], top_k=3)
    assert len(out) == 1 and len(out[0]) == 3
    scores = [s for _, s in out[0]]
    assert scores == sorted(scores, reverse=True)
```

`tests/test_plugin.py`:
```python
import pytest

sragents_base = pytest.importorskip("sragents.retrieve.base")


def test_factories_registered_and_accept_string_kwargs(tmp_path, monkeypatch):
    import hyskill.plugin  # noqa: F401  (import side-effect registers)
    names = sragents_base.list_retrievers()
    assert "hyskill" in names and "naive_hyde" in names
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_naive_hyde.py -q`
Expected: FAIL

- [ ] **Step 3: Implement `hyskill/naive_hyde.py`**

```python
"""Naive HyDE baseline: unstructured passage generation, single-vector retrieval.

Faithful port of HyDE (Gao et al., ACL 2023) to the skill corpus: average the
K passage embeddings with the query embedding (eq. 8) and rank the full-text
corpus index by inner product. No field structure, no BM25, no fusion.
"""

import numpy as np

from hyskill.embedder import Embedder


class NaiveHydeRetriever:
    def __init__(self, generator, st_model=None,
                 encoder_name: str = "BAAI/bge-base-en-v1.5"):
        self._generator = generator
        self._embedder = Embedder(model=st_model, model_name=encoder_name)

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self._ids = list(corpus_ids)
        self._emb = self._embedder.encode(list(corpus_texts))

    def retrieve(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        results = []
        q_vecs = self._embedder.encode(list(queries))
        for query, q_vec in zip(queries, q_vecs):
            docs = self._generator.generate(query)
            vecs = [q_vec]
            if docs:
                vecs = list(self._embedder.encode(docs)) + [q_vec]
            v = np.mean(vecs, axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            scores = self._emb @ v
            order = np.argsort(-scores)[:top_k]
            results.append([(self._ids[j], float(scores[j])) for j in order])
        return results
```

- [ ] **Step 4: Implement `hyskill/plugin.py`**

```python
"""SR-Agents plugin: registers `hyskill` and `naive_hyde` retrievers.

Usage:
    sragents --plugin hyskill.plugin retrieve --retriever hyskill \
        --retriever-arg corpus_path=data/bench/corpus/corpus.json \
        --retriever-arg model=Qwen/Qwen3-4B-Instruct \
        --retriever-arg api_base=http://localhost:8000/v1 \
        --retriever-arg k_samples=4 \
        --retriever-arg cache_dir=results/hyp_cache ...

All --retriever-arg values arrive as strings; factories coerce types.
`mock_generator=1` swaps in an offline echo generator (integration tests).
"""

from sragents.retrieve.base import register

from hyskill.generator import (HypotheticalGenerator, OpenAIClient,
                               PASSAGE_TEMPLATE, SKILL_TEMPLATE)
from hyskill.naive_hyde import NaiveHydeRetriever
from hyskill.retriever import HySkillRetriever


class _MockClient:
    def complete(self, prompt: str, temperature: float) -> str:
        task = prompt.rsplit("Task:", 1)[-1].strip()[:120]
        return ("---\nname: hypothetical-skill\ndescription: " + task +
                "\n---\n1. analyse the task\n2. apply the method\n\n"
                "```python\nsolve()\n```")


def _generator(template, model="", api_base="", k_samples="4",
               temperature="0.7", cache_dir="", mock_generator="0"):
    client = (_MockClient() if str(mock_generator) == "1"
              else OpenAIClient(model=model, api_base=api_base))
    return HypotheticalGenerator(
        client=client, k_samples=int(k_samples), temperature=float(temperature),
        template=template, cache_dir=cache_dir or None,
        model_tag=f"{model}|{template[:20]}")


@register("hyskill")
def hyskill_factory(corpus_path, encoder_name="BAAI/bge-base-en-v1.5",
                    rrf_k="60", **gen_kwargs):
    return HySkillRetriever(
        corpus_path=corpus_path,
        generator=_generator(SKILL_TEMPLATE, **gen_kwargs),
        encoder_name=encoder_name, rrf_k=int(rrf_k))


@register("naive_hyde")
def naive_hyde_factory(encoder_name="BAAI/bge-base-en-v1.5", **gen_kwargs):
    return NaiveHydeRetriever(
        generator=_generator(PASSAGE_TEMPLATE, **gen_kwargs),
        encoder_name=encoder_name)
```

- [ ] **Step 5: Install SR-Agents into the venv, run both test files**

Run: `.venv/bin/pip install -q -e external/SR-Agents && .venv/bin/pytest tests/test_naive_hyde.py tests/test_plugin.py -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add hyskill/naive_hyde.py hyskill/plugin.py tests/test_naive_hyde.py tests/test_plugin.py
git commit -m "feat: naive-HyDE baseline and SR-Agents plugin registration"
```

---

### Task 8: End-to-end CLI smoke test (mock generator, tiny encoder)

**Files:**
- Create: `tests/fixtures/tiny_instances.json`
- Create: `scripts/smoke.sh`

- [ ] **Step 1: Write `tests/fixtures/tiny_instances.json`**

```json
[
  {"instance_id": "t_000", "dataset": "theoremqa", "question": "How many ways to split 8 items into 5 ordered lists?", "skill_annotations": ["s_lah"], "eval_data": {"answer": "11760", "answer_type": "integer"}},
  {"instance_id": "t_001", "dataset": "theoremqa", "question": "Convert this scanned PDF report into a Markdown table.", "skill_annotations": ["s_pdf"], "eval_data": {"answer": "n/a", "answer_type": "string"}}
]
```

- [ ] **Step 2: Write `scripts/smoke.sh`**

```bash
#!/usr/bin/env bash
# End-to-end Stage-1 smoke test: plugin loads, retrieval runs, schema written.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(mktemp -d)
.venv/bin/python -m sragents.cli.main --plugin hyskill.plugin retrieve \
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
```

Note: if `python -m sragents.cli.main` is not the entry module, check `external/SR-Agents/pyproject.toml` `[project.scripts]` and use the installed `sragents` console script from `.venv/bin/sragents` instead.

- [ ] **Step 3: Install runtime deps and run smoke**

Run: `.venv/bin/pip install -q sentence-transformers && chmod +x scripts/smoke.sh && ./scripts/smoke.sh`
Expected: `SMOKE OK` with a metrics dict (downloads MiniLM ~90MB on first run).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/tiny_instances.json scripts/smoke.sh
git commit -m "test: end-to-end CLI smoke test with mock generator"
```

---

### Task 9: Phase 0 experiment scripts

**Files:**
- Create: `scripts/run_phase0.sh`
- Create: `scripts/analyze.py`

- [ ] **Step 1: Write `scripts/run_phase0.sh`**

```bash
#!/usr/bin/env bash
# Phase 0: retrieval-stage comparison on SRA-Bench (5 datasets, ToolQA deferred).
# Usage: MODEL=... API_BASE=... [PILOT=1] ./scripts/run_phase0.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SRA=external/SR-Agents
CORPUS=$SRA/data/bench/corpus/corpus.json
DATASETS=(theoremqa logicbench medcalcbench champ bigcodebench)
mkdir -p results/retrieval results/hyp_cache

for DS in "${DATASETS[@]}"; do
  INST=$SRA/data/bench/instances/$DS.json
  if [[ "${PILOT:-0}" == "1" ]]; then
    .venv/bin/python -c "
import json; d=json.load(open('$INST')); json.dump(d[:20], open('results/pilot_$DS.json','w'))"
    INST=results/pilot_$DS.json
  fi
  for R in bm25 bge hybrid; do
    .venv/bin/sragents retrieve --retriever $R \
      --corpus $CORPUS --instances $INST \
      --output results/retrieval/$DS-$R.json --top-k 50
  done
  .venv/bin/sragents --plugin hyskill.plugin retrieve --retriever naive_hyde \
    --retriever-arg model="$MODEL" --retriever-arg api_base="$API_BASE" \
    --retriever-arg k_samples=4 --retriever-arg cache_dir=results/hyp_cache \
    --corpus $CORPUS --instances $INST \
    --output results/retrieval/$DS-naive_hyde.json --top-k 50
  .venv/bin/sragents --plugin hyskill.plugin retrieve --retriever hyskill \
    --retriever-arg corpus_path=$CORPUS \
    --retriever-arg model="$MODEL" --retriever-arg api_base="$API_BASE" \
    --retriever-arg k_samples=4 --retriever-arg cache_dir=results/hyp_cache \
    --corpus $CORPUS --instances $INST \
    --output results/retrieval/$DS-hyskill.json --top-k 50
done
.venv/bin/python scripts/analyze.py results/retrieval
```

- [ ] **Step 2: Write `scripts/analyze.py`**

```python
"""Aggregate SR-Agents retrieval JSONs into a Markdown comparison table."""

import json
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "nDCG@10"]


def main(result_dir: str) -> None:
    table = defaultdict(dict)  # (dataset, retriever) -> metrics
    for p in sorted(Path(result_dir).glob("*.json")):
        d = json.loads(p.read_text())
        meta, metrics = d.get("metadata", {}), d.get("metrics") or {}
        table[(meta.get("dataset", p.stem), meta.get("retriever", "?"))] = metrics
    datasets = sorted({k[0] for k in table})
    retrievers = sorted({k[1] for k in table})
    for ds in datasets:
        print(f"\n## {ds}\n")
        print("| retriever | " + " | ".join(METRICS) + " |")
        print("|" + "---|" * (len(METRICS) + 1))
        for r in retrievers:
            m = table.get((ds, r), {})
            row = " | ".join(f"{m.get(x, float('nan')):.3f}" if x in m else "—"
                             for x in METRICS)
            print(f"| {r} | {row} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/retrieval")
```

Note: metric key names must match what `sragents.retrieve.metrics` emits — verify with one smoke-run JSON and adjust `METRICS` if the emitted keys differ (e.g. `recall@10`).

- [ ] **Step 3: Make executable, commit**

```bash
chmod +x scripts/run_phase0.sh
git add scripts/run_phase0.sh scripts/analyze.py
git commit -m "feat: Phase 0 batch runner and results aggregator"
```

---

### Task 10: Pilot run (execution checklist, requires live endpoint + GPU)

- [ ] **Step 1: Serve or configure a generator endpoint**

Local (GPU): `vllm serve Qwen/Qwen3-4B-Instruct --port 8000` → `MODEL=Qwen/Qwen3-4B-Instruct API_BASE=http://localhost:8000/v1`; or any hosted OpenAI-compatible endpoint + `OPENAI_API_KEY`.

- [ ] **Step 2: Pilot (20 instances × 5 datasets)**

Run: `MODEL=... API_BASE=... PILOT=1 ./scripts/run_phase0.sh`
Expected: per-dataset Markdown tables; sanity: bge ≫ random, hyskill runs without fallback warnings (check `results/hyp_cache` fills up).

- [ ] **Step 3: Inspect 5 cached hypothetical skills manually**

Run: `ls results/hyp_cache | head -5` then read them — verify frontmatter + steps + code skeleton shape. If malformed, tune `SKILL_TEMPLATE`, bump `model_tag`, re-run pilot.

- [ ] **Step 4: Full run + commit results tables**

Run: `MODEL=... API_BASE=... ./scripts/run_phase0.sh | tee results/phase0_tables.md`
Then: `git add results/phase0_tables.md -f && git commit -m "exp: Phase 0 retrieval comparison on SRA-Bench"`

Go/no-go: hyskill > {bge, hybrid, naive_hyde} on Recall@10/nDCG@10 across ≥3 datasets → proceed to ablations & arXiv draft; hyskill ≈ naive_hyde → structured generation not pulling weight, revisit template/fields before scaling spend.
