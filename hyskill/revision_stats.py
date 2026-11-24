"""Paired hierarchical bootstrap for the IJCAI-27 revision analyses.

The resampling order matches the runtime-matched baseline estimator: models,
then domains, then paired instances inside a cell. Cell means are averaged
without instance weights.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

import numpy as np

from hyskill.k_ablation import DOMAINS, validation_ids


PairArrays: TypeAlias = tuple[np.ndarray, np.ndarray]
CellKey: TypeAlias = tuple[str, str]
IdFilter: TypeAlias = Callable[[str, str], frozenset[str]]


def comparison_seed(base_seed: int, key: str) -> int:
    """Derive an order-independent NumPy seed for one named comparison."""

    digest: bytes = hashlib.sha256(key.encode()).digest()
    return (int.from_bytes(digest[:8], byteorder="big") ^ base_seed) % (2**63)


def test_ids(domain_ids: frozenset[str]) -> frozenset[str]:
    """Return the held-out ids for one domain's full id set."""

    if not domain_ids:
        raise ValueError("domain id set is empty")
    return domain_ids - validation_ids(domain_ids)


def paired_cells(
    rows_a: Sequence[Mapping[str, object]],
    rows_b: Sequence[Mapping[str, object]],
    metric: str,
    ids: IdFilter,
) -> dict[CellKey, PairArrays]:
    """Align one metric into per-cell arrays.

    ``ids(model, domain)`` selects the instance ids that belong in that cell.
    Rows outside the selection are ignored. The two sides of a cell must
    contain the same selected ids.
    """

    left = _index_rows(rows_a, metric, "a")
    right = _index_rows(rows_b, metric, "b")
    cells: dict[CellKey, PairArrays] = {}
    for model, domain in sorted(set(left) | set(right)):
        allowed = ids(model, domain)
        if not isinstance(allowed, frozenset):
            raise ValueError(
                "id filter must return a frozenset: "
                f"model={model}, domain={domain}, type={type(allowed).__name__}"
            )
        left_ids = {
            instance_id: value
            for instance_id, value in left.get((model, domain), {}).items()
            if instance_id in allowed
        }
        right_ids = {
            instance_id: value
            for instance_id, value in right.get((model, domain), {}).items()
            if instance_id in allowed
        }
        if set(left_ids) != set(right_ids):
            missing_a = sorted(set(right_ids) - set(left_ids))[:8]
            missing_b = sorted(set(left_ids) - set(right_ids))[:8]
            raise ValueError(
                "Paired cells have different instance ids: "
                f"model={model}, domain={domain}, metric={metric}, "
                f"missing_a={missing_a}, missing_b={missing_b}"
            )
        if not left_ids:
            continue
        ordered = sorted(left_ids)
        cells[(model, domain)] = (
            np.asarray([left_ids[instance_id] for instance_id in ordered], dtype=np.float64),
            np.asarray([right_ids[instance_id] for instance_id in ordered], dtype=np.float64),
        )
    if not cells:
        raise ValueError(f"paired cells are empty: metric={metric}")
    return cells


def hierarchical_paired_bootstrap(
    cells: Mapping[CellKey, PairArrays],
    samples: int,
    seed: int,
    name: str,
) -> dict[str, object]:
    """Resample models, domains, then paired instances within each cell."""

    if samples <= 0:
        raise ValueError(f"Bootstrap sample count must be positive: value={samples}")
    if not cells:
        raise ValueError(f"Bootstrap cells are empty: name={name}")
    models = sorted({model for model, _domain in cells})
    domains = _domain_order(cells)
    _require_rectangle(cells, models, domains)
    model_count = len(models)
    domain_count = len(domains)
    cell_distributions = np.empty((model_count, domain_count, samples), dtype=np.float64)
    means_a = np.empty((model_count, domain_count), dtype=np.float64)
    means_b = np.empty_like(means_a)
    total_instances = 0
    for model_index, model in enumerate(models):
        for domain_index, domain in enumerate(domains):
            values_a, values_b = cells[(model, domain)]
            if values_a.shape != values_b.shape or values_a.ndim != 1 or values_a.size == 0:
                raise ValueError(
                    "Paired cell must be two non-empty vectors of equal length: "
                    f"model={model}, domain={domain}, "
                    f"a={values_a.shape}, b={values_b.shape}"
                )
            differences = values_a - values_b
            means_a[model_index, domain_index] = values_a.mean()
            means_b[model_index, domain_index] = values_b.mean()
            total_instances += int(values_a.size)
            cell_distributions[model_index, domain_index, :] = _bootstrap_distribution(
                differences,
                samples,
                comparison_seed(seed, f"{name}:{model}:{domain}:instances"),
            )
    hierarchy = np.random.default_rng(comparison_seed(seed, f"{name}:hierarchy"))
    model_indices = hierarchy.integers(0, model_count, size=(samples, model_count), endpoint=False)
    domain_indices = hierarchy.integers(
        0,
        domain_count,
        size=(samples, model_count, domain_count),
        endpoint=False,
    )
    sampled_models = np.broadcast_to(model_indices[:, :, None], domain_indices.shape)
    bootstrap_indices = np.broadcast_to(np.arange(samples)[:, None, None], domain_indices.shape)
    distribution = cell_distributions[sampled_models, domain_indices, bootstrap_indices].mean(axis=(1, 2))
    lower, upper = np.percentile(distribution, [2.5, 97.5])
    probability_nonpositive = float(np.mean(distribution <= 0.0))
    probability_nonnegative = float(np.mean(distribution >= 0.0))
    return {
        "mean_a": float(means_a.mean()),
        "mean_b": float(means_b.mean()),
        "difference": float((means_a - means_b).mean()),
        "ci95": [float(lower), float(upper)],
        "p_two_sided": min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)),
        "n_models": model_count,
        "n_domains": domain_count,
        "n_instances": total_instances,
        "samples": samples,
        "seed": seed,
    }


def _domain_order(cells: Mapping[CellKey, PairArrays]) -> list[str]:
    present = {domain for _model, domain in cells}
    ordered = [domain for domain in DOMAINS if domain in present]
    ordered.extend(sorted(present.difference(DOMAINS)))
    return ordered


def _require_rectangle(
    cells: Mapping[CellKey, PairArrays],
    models: Sequence[str],
    domains: Sequence[str],
) -> None:
    expected = {(model, domain) for model in models for domain in domains}
    if set(cells) != expected:
        missing = sorted(expected - set(cells))[:8]
        extra = sorted(set(cells) - expected)[:8]
        raise ValueError(
            "Hierarchical bootstrap requires a rectangular model-domain grid: "
            f"missing={missing}, extra={extra}"
        )


def _bootstrap_distribution(differences: np.ndarray, samples: int, seed: int) -> np.ndarray:
    """Return paired instance-bootstrap means.

    Chunk size matches the runtime-matched baseline estimator, because NumPy's
    generator stream depends on how many integers each call draws.
    """

    if differences.ndim != 1 or differences.size == 0:
        raise ValueError(
            f"Bootstrap differences must be a non-empty vector: shape={differences.shape}"
        )
    generator = np.random.default_rng(seed)
    sample_count = int(differences.size)
    chunk_size = max(1, min(256, 4_000_000 // sample_count))
    distribution = np.empty(samples, dtype=np.float64)
    offset = 0
    while offset < samples:
        current = min(chunk_size, samples - offset)
        indices = generator.integers(0, sample_count, size=(current, sample_count), endpoint=False)
        distribution[offset : offset + current] = differences[indices].mean(axis=1)
        offset += current
    return distribution


def _index_rows(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    side: str,
) -> dict[CellKey, dict[str, float]]:
    grouped: dict[CellKey, dict[str, float]] = {}
    for index, row in enumerate(rows):
        model = _require_text(row.get("model"), f"{side}[{index}].model")
        domain = _require_text(row.get("domain"), f"{side}[{index}].domain")
        instance_id = _require_text(row.get("instance_id"), f"{side}[{index}].instance_id")
        if metric not in row:
            raise ValueError(f"metric is missing: side={side}, index={index}, metric={metric}")
        try:
            value = float(row[metric])  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"metric is not numeric: side={side}, index={index}, metric={metric}"
            ) from error
        cell = grouped.setdefault((model, domain), {})
        if instance_id in cell:
            raise ValueError(
                "duplicate instance in one side: "
                f"side={side}, model={model}, domain={domain}, instance_id={instance_id}"
            )
        cell[instance_id] = value
    return grouped


def _require_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string: value={value!r}")
    return value
