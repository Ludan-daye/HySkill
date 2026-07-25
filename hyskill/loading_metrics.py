"""Pure decision-level loading metrics for the K=2 downstream study."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

from hyskill.downstream_reuse import FailureCategory


LoadingArm = Literal["routed_always", "routed_gated", "routed_select"]


class LoadingDecisionRow(TypedDict):
    """One expected loading decision before answer execution."""

    schema_version: str
    instance_id: str
    model: str
    domain: str
    arm: LoadingArm
    expected_skill_ids: list[str]
    gold_skill_ids: list[str]
    loaded: bool
    hit: bool | None
    gold_loaded: bool
    is_validation: bool
    failure_category: FailureCategory
    decision_source_sha256: str


class LoadingMetrics(TypedDict):
    """Count-complete loading metrics for one explicit support set."""

    instances: int
    loaded: int
    gold_loaded: int
    method_failures: int
    loaded_skill_precision: float | None
    loading_rate: float
    gold_load_rate: float
    selection_failure_rate: float


def compute_loading_metrics(
    rows: Sequence[LoadingDecisionRow],
) -> LoadingMetrics:
    """Compute the frozen H/L, L/N, and H/N loading metrics."""

    if not rows:
        raise ValueError("Loading metric support must contain at least one row")
    instances: int = len(rows)
    loaded: int = sum(1 for row in rows if row["loaded"])
    gold_loaded: int = sum(1 for row in rows if row["gold_loaded"])
    method_failures: int = sum(
        1 for row in rows if row["failure_category"] == "method_failure"
    )
    return {
        "instances": instances,
        "loaded": loaded,
        "gold_loaded": gold_loaded,
        "method_failures": method_failures,
        "loaded_skill_precision": (
            gold_loaded / loaded if loaded > 0 else None
        ),
        "loading_rate": loaded / instances,
        "gold_load_rate": gold_loaded / instances,
        "selection_failure_rate": method_failures / instances,
    }


def mean_defined(values: Sequence[float | None]) -> tuple[float | None, int]:
    """Average defined values and report the contributing denominator."""

    defined: list[float] = [value for value in values if value is not None]
    if not defined:
        return None, 0
    return sum(defined) / len(defined), len(defined)
