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
