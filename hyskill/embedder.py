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
