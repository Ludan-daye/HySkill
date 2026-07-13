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
