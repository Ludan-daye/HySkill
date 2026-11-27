"""Generate the fixed-selector candidate-source ablation figure."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from publication_figure_style import configure_style


class ModelSpec(TypedDict):
    slug: str
    label: str
    result_tag: str


class Counts(TypedDict):
    correct: int
    total: int


class ModelResult(TypedDict):
    slug: str
    label: str
    instance_count: int
    bm25_correct: int
    hyskill_correct: int
    bm25_accuracy: float
    hyskill_accuracy: float
    delta: float


class PlotRow(TypedDict):
    label: str
    bm25_accuracy: float
    hyskill_accuracy: float
    delta: float
    pooled: bool


class BarStyle(TypedDict):
    edgecolor: str
    facecolor: str
    hatch: str


class OutputFiles(TypedDict):
    pdf: str
    png: str
    grayscale_png: str


class Manifest(TypedDict):
    comparison: str
    chart_type: str
    metric: str
    domains: tuple[str, ...]
    selector_control: str
    display_rounding: str
    candidate_sources: dict[str, str]
    model_sources: dict[str, tuple[str, ...]]
    models: tuple[ModelResult, ...]
    pooled: ModelResult
    outputs: OutputFiles
    caption_draft: str
    alt_text: str


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
HELD_OUT_PER_MODEL = 2265
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "paper" / "figures" / "select-candidate-source-ablation"
)
MODEL_SPECS: tuple[ModelSpec, ...] = (
    {
        "slug": "qwen35-4b",
        "label": "Qwen3.5-4B",
        "result_tag": "qwen3.5-4b-reference",
    },
    {
        "slug": "qwen35-9b",
        "label": "Qwen3.5-9B",
        "result_tag": "qwen35-9b",
    },
    {
        "slug": "glm4-9b",
        "label": "GLM-4-9B",
        "result_tag": "glm4-9b",
    },
    {
        "slug": "llama31-8b",
        "label": "Llama-3.1-8B",
        "result_tag": "llama31-8b",
    },
    {
        "slug": "mistral7b",
        "label": "Mistral-7B",
        "result_tag": "mistral7b",
    },
)
K2_ANSWER_SCHEMA = "k2-public-answer-row-v1"
BASELINE_ANSWER_SCHEMA = "runtime-matched-public-answer-row-v1"
DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
)
COLOR_STYLES: dict[str, BarStyle] = {
    "bm25": {
        "edgecolor": "#626B73",
        "facecolor": "#D9DEE3",
        "hatch": "////",
    },
    "hyskill": {
        "edgecolor": "#294E75",
        "facecolor": "#4778AE",
        "hatch": "\\\\\\\\",
    },
}
GRAYSCALE_STYLES: dict[str, BarStyle] = {
    "bm25": {
        "edgecolor": "#5E5E5E",
        "facecolor": "white",
        "hatch": "////",
    },
    "hyskill": {
        "edgecolor": "#1F1F1F",
        "facecolor": "#666666",
        "hatch": "\\\\\\\\",
    },
}
PLOT_LABELS: dict[str, str] = {
    "qwen35-4b": "Qwen\n3.5-4B",
    "qwen35-9b": "Qwen\n3.5-9B",
    "glm4-9b": "GLM\n4-9B",
    "llama31-8b": "Llama\n3.1-8B",
    "mistral7b": "Mistral\n7B",
    "pooled-five-models": "5-model\npooled",
}


def require_object(value: object, context: str) -> dict[str, object]:
    """Return one decoded JSON object or raise a contextual schema error."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object for {context}, observed={type(value)}")
    return cast(dict[str, object], value)


def require_string(record: dict[str, object], key: str, context: str) -> str:
    """Read one required string field from a decoded record."""
    value = record.get(key)
    if not isinstance(value, str):
        raise TypeError(
            f"Expected string field for {context}: key={key}, observed={value!r}"
        )
    return value


def require_boolean(record: dict[str, object], key: str, context: str) -> bool:
    """Read one required Boolean field from a decoded record."""
    value = record.get(key)
    if type(value) is not bool:
        raise TypeError(
            f"Expected Boolean field for {context}: key={key}, observed={value!r}"
        )
    return cast(bool, value)


def empty_domain_counts() -> dict[str, Counts]:
    """Create zeroed counters for the four rule-evaluable domains."""
    return {domain: {"correct": 0, "total": 0} for domain in DOMAINS}


def k2_answer_path(spec: ModelSpec) -> Path:
    """Return one model's active K=2 public answer archive."""
    return (
        REPOSITORY_ROOT
        / "community-results"
        / spec["result_tag"]
        / "k2"
        / "answer_per_instance.jsonl.gz"
    )


def baseline_answer_path(spec: ModelSpec) -> Path:
    """Return one model's fresh runtime-matched baseline archive."""
    return (
        REPOSITORY_ROOT
        / "community-results"
        / spec["result_tag"]
        / "baselines-runtime-matched"
        / "answer_per_instance.jsonl.gz"
    )


def load_public_answer_counts(
    path: Path,
    schema_version: str,
    model: str,
    arm: str,
) -> dict[str, Counts]:
    """Count one arm's public per-instance correctness by domain."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing public answer archive: {path}")

    counts = empty_domain_counts()
    seen_keys: set[tuple[str, str]] = set()
    with gzip.open(path, mode="rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            context = f"path={path}, line={line_number}"
            record = require_object(json.loads(line), context)
            if require_string(record, "schema_version", context) != schema_version:
                raise ValueError(
                    f"Unexpected public answer schema: {context}, "
                    f"expected={schema_version}, "
                    f"observed={record.get('schema_version')!r}"
                )
            if require_string(record, "model", context) != model:
                raise ValueError(
                    f"Unexpected result tag: {context}, expected={model}, "
                    f"observed={record.get('model')!r}"
                )
            if require_string(record, "arm", context) != arm:
                continue
            instance_id = require_string(record, "instance_id", context)
            domain = require_string(record, "domain", context)
            if domain not in DOMAINS:
                raise ValueError(
                    f"Unexpected domain in gating archive: {context}, domain={domain}"
                )
            instance_key = (domain, instance_id)
            if instance_key in seen_keys:
                raise ValueError(
                    f"Duplicate gating record: {context}, domain={domain}, "
                    f"instance_id={instance_id}"
                )
            seen_keys.add(instance_key)
            # Held-out only: the router and gate were selected on the 20% validation split.
            if require_boolean(record, "is_validation", context):
                continue
            correct = require_boolean(record, "correct", context)
            counts[domain]["total"] += 1
            counts[domain]["correct"] += int(correct)
    observed_domains = {
        domain
        for domain, domain_counts in counts.items()
        if domain_counts["total"] > 0
    }
    if observed_domains != set(DOMAINS):
        raise ValueError(
            f"Unexpected domain coverage in public answers: path={path}, "
            f"expected={DOMAINS}, observed={tuple(sorted(observed_domains))}"
        )
    pooled_total = sum(counts[domain]["total"] for domain in DOMAINS)
    if pooled_total != HELD_OUT_PER_MODEL:
        raise ValueError(
            f"Unexpected arm denominator: path={path}, arm={arm}, "
            f"expected={HELD_OUT_PER_MODEL}, observed={pooled_total}"
        )
    return counts


def sum_counts(domain_counts: dict[str, Counts]) -> Counts:
    """Pool exact correct and total counts across the declared domains."""
    missing_domains = set(DOMAINS) - set(domain_counts)
    if missing_domains:
        raise ValueError(
            f"Cannot pool incomplete domain counts: missing={tuple(sorted(missing_domains))}"
        )
    return {
        "correct": sum(domain_counts[domain]["correct"] for domain in DOMAINS),
        "total": sum(domain_counts[domain]["total"] for domain in DOMAINS),
    }


def build_model_result(
    slug: str,
    label: str,
    bm25_domain_counts: dict[str, Counts],
    hyskill_domain_counts: dict[str, Counts],
) -> ModelResult:
    """Build one paired model result after validating matched denominators."""
    bm25_counts = sum_counts(bm25_domain_counts)
    hyskill_counts = sum_counts(hyskill_domain_counts)
    for domain in DOMAINS:
        bm25_total = bm25_domain_counts[domain]["total"]
        hyskill_total = hyskill_domain_counts[domain]["total"]
        if bm25_total != hyskill_total:
            raise ValueError(
                f"Mismatched candidate-source denominators: model={label}, "
                f"domain={domain}, bm25_total={bm25_total}, "
                f"hyskill_total={hyskill_total}"
            )
    if bm25_counts["total"] != hyskill_counts["total"]:
        raise ValueError(
            f"Mismatched pooled denominators: model={label}, "
            f"bm25_total={bm25_counts['total']}, "
            f"hyskill_total={hyskill_counts['total']}"
        )

    bm25_accuracy = 100.0 * bm25_counts["correct"] / bm25_counts["total"]
    hyskill_accuracy = 100.0 * hyskill_counts["correct"] / hyskill_counts["total"]
    return {
        "slug": slug,
        "label": label,
        "instance_count": bm25_counts["total"],
        "bm25_correct": bm25_counts["correct"],
        "hyskill_correct": hyskill_counts["correct"],
        "bm25_accuracy": bm25_accuracy,
        "hyskill_accuracy": hyskill_accuracy,
        "delta": hyskill_accuracy - bm25_accuracy,
    }


def build_result(spec: ModelSpec) -> ModelResult:
    """Build one matched candidate-source pair from active public packs."""
    bm25_counts = load_public_answer_counts(
        baseline_answer_path(spec),
        BASELINE_ANSWER_SCHEMA,
        spec["result_tag"],
        "select_bm25",
    )
    hyskill_counts = load_public_answer_counts(
        k2_answer_path(spec),
        K2_ANSWER_SCHEMA,
        spec["result_tag"],
        "routed_select",
    )
    return build_model_result(
        spec["slug"],
        spec["label"],
        bm25_counts,
        hyskill_counts,
    )


def build_pooled_result(model_results: tuple[ModelResult, ...]) -> ModelResult:
    """Pool exact decisions across all eligible models."""
    if not model_results:
        raise ValueError("Cannot pool an empty model result sequence")
    bm25_correct = sum(result["bm25_correct"] for result in model_results)
    hyskill_correct = sum(result["hyskill_correct"] for result in model_results)
    instance_count = sum(result["instance_count"] for result in model_results)
    bm25_accuracy = 100.0 * bm25_correct / instance_count
    hyskill_accuracy = 100.0 * hyskill_correct / instance_count
    return {
        "slug": "pooled-five-models",
        "label": "5-model pooled",
        "instance_count": instance_count,
        "bm25_correct": bm25_correct,
        "hyskill_correct": hyskill_correct,
        "bm25_accuracy": bm25_accuracy,
        "hyskill_accuracy": hyskill_accuracy,
        "delta": hyskill_accuracy - bm25_accuracy,
    }


def build_plot_rows(
    model_results: tuple[ModelResult, ...],
    pooled_result: ModelResult,
) -> tuple[PlotRow, ...]:
    """Adapt model and pooled results into display rows."""
    model_rows = tuple(
        {
            "label": PLOT_LABELS[result["slug"]],
            "bm25_accuracy": result["bm25_accuracy"],
            "hyskill_accuracy": result["hyskill_accuracy"],
            "delta": result["delta"],
            "pooled": False,
        }
        for result in model_results
    )
    pooled_row: PlotRow = {
        "label": PLOT_LABELS[pooled_result["slug"]],
        "bm25_accuracy": pooled_result["bm25_accuracy"],
        "hyskill_accuracy": pooled_result["hyskill_accuracy"],
        "delta": pooled_result["delta"],
        "pooled": True,
    }
    return model_rows + (pooled_row,)


def draw_chart(
    rows: tuple[PlotRow, ...],
    styles: dict[str, BarStyle],
) -> Figure:
    """Draw one single-column grouped bar chart."""
    if not rows:
        raise ValueError("Cannot draw an empty candidate-source comparison")

    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=(3.32, 2.65), dpi=300)
    positions = tuple(float(index) for index in range(len(rows) - 1)) + (
        float(len(rows) - 1) + 0.35,
    )
    bar_width = 0.34
    bar_offset = 0.19
    bm25_style = styles["bm25"]
    hyskill_style = styles["hyskill"]
    bm25_positions = tuple(position - bar_offset for position in positions)
    hyskill_positions = tuple(position + bar_offset for position in positions)
    bm25_values = tuple(row["bm25_accuracy"] for row in rows)
    hyskill_values = tuple(row["hyskill_accuracy"] for row in rows)

    bm25_bars = axes.bar(
        bm25_positions,
        bm25_values,
        width=bar_width,
        color=bm25_style["facecolor"],
        edgecolor=bm25_style["edgecolor"],
        linewidth=0.7,
        hatch=bm25_style["hatch"],
        label="BM25 top-50",
        zorder=3,
    )
    hyskill_bars = axes.bar(
        hyskill_positions,
        hyskill_values,
        width=bar_width,
        color=hyskill_style["facecolor"],
        edgecolor=hyskill_style["edgecolor"],
        linewidth=0.7,
        hatch=hyskill_style["hatch"],
        label="HySkill top-50",
        zorder=3,
    )
    bm25_bars[-1].set_linewidth(1.05)
    hyskill_bars[-1].set_linewidth(1.05)

    for index, row in enumerate(rows):
        displayed_bm25 = float(f"{row['bm25_accuracy']:.1f}")
        displayed_hyskill = float(f"{row['hyskill_accuracy']:.1f}")
        displayed_delta = displayed_hyskill - displayed_bm25
        axes.text(
            positions[index],
            max(row["bm25_accuracy"], row["hyskill_accuracy"]) + 2.2,
            f"{displayed_delta:+.1f}",
            ha="center",
            va="bottom",
            color=hyskill_style["edgecolor"],
            fontsize=6.5,
            fontweight="bold",
            zorder=5,
        )

    separator = (positions[-2] + positions[-1]) / 2.0
    axes.axvline(
        separator,
        color="#B7BEC5",
        linewidth=0.7,
        linestyle=(0, (2.0, 2.0)),
        zorder=1,
    )
    axes.set_xlim(positions[0] - 0.55, positions[-1] + 0.55)
    axes.set_ylim(0.0, 86.0)
    axes.set_yticks((0.0, 20.0, 40.0, 60.0, 80.0))
    axes.set_ylabel("End-to-End Accuracy (%)")
    axes.set_xticks(positions, tuple(row["label"] for row in rows))
    axes.set_axisbelow(True)
    axes.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.95, zorder=0)
    axes.grid(axis="x", visible=False)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color("#30343B")
    axes.spines["bottom"].set_color("#30343B")
    axes.spines["left"].set_linewidth(0.85)
    axes.spines["bottom"].set_linewidth(0.85)
    axes.tick_params(axis="x", length=0.0, pad=4.0)
    axes.tick_params(axis="y", color="#30343B", width=0.8, length=2.5)
    axes.get_xticklabels()[-1].set_fontweight("bold")
    figure.legend(
        handles=(bm25_bars[0], hyskill_bars[0]),
        labels=("BM25 top-50", "HySkill top-50"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
        fontsize=7.5,
    )
    figure.subplots_adjust(left=0.18, right=0.99, bottom=0.24, top=0.84)
    return figure


def save_outputs(
    color_figure: Figure,
    grayscale_figure: Figure,
) -> OutputFiles:
    """Save vector artwork plus color and grayscale review previews."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIRECTORY / "select-candidate-source-ablation.pdf"
    png_path = OUTPUT_DIRECTORY / "select-candidate-source-ablation.png"
    grayscale_path = (
        OUTPUT_DIRECTORY / "select-candidate-source-ablation-grayscale.png"
    )
    color_figure.savefig(pdf_path, format="pdf", facecolor="white")
    color_figure.savefig(png_path, format="png", dpi=300, facecolor="white")
    grayscale_figure.savefig(
        grayscale_path,
        format="png",
        dpi=300,
        facecolor="white",
    )
    return {
        "pdf": str(pdf_path.relative_to(REPOSITORY_ROOT)),
        "png": str(png_path.relative_to(REPOSITORY_ROOT)),
        "grayscale_png": str(grayscale_path.relative_to(REPOSITORY_ROOT)),
    }


def source_manifest() -> dict[str, tuple[str, ...]]:
    """Record every analysis-pack source used to calculate plotted values."""
    return {
        spec["slug"]: (
            str(
                baseline_answer_path(spec).relative_to(REPOSITORY_ROOT)
            ),
            str(k2_answer_path(spec).relative_to(REPOSITORY_ROOT)),
        )
        for spec in MODEL_SPECS
    }


def build_manifest(
    model_results: tuple[ModelResult, ...],
    pooled_result: ModelResult,
    outputs: OutputFiles,
) -> Manifest:
    """Describe the comparison, exact counts, sources, and publication text."""
    return {
        "comparison": (
            "Hold the selector and answering model fixed and replace BM25 "
            "top-50 candidates with routed K=2 HySkill top-50 candidates"
        ),
        "chart_type": "Single-panel grouped bar chart",
        "metric": "Held-out end-to-end task accuracy pooled over four rule-evaluable domains",
        "domains": DOMAINS,
        "selector_control": (
            "The selector and answering model are identical within each model pair"
        ),
        "display_rounding": (
            "Gain labels subtract one-decimal rounded endpoint accuracies, while "
            "exact gains remain in the model records"
        ),
        "candidate_sources": {
            "bm25": "Fresh runtime-matched BM25 top-50 candidate pool",
            "hyskill": (
                "Active routed HySkill K=2 top-50 candidate pool for all five "
                "eligible models"
            ),
        },
        "model_sources": source_manifest(),
        "models": model_results,
        "pooled": pooled_result,
        "outputs": outputs,
        "caption_draft": (
            "Effect of HySkill candidate construction under a fixed LLM selector. "
            "Each model pools 2,265 held-out instances from four rule-evaluable domains. "
            "Within each pair, the selector and answering model are held fixed; "
            "only the top-50 candidate source changes from native BM25 to HySkill. "
            "All five HySkill arms use active routed K=2 candidates, while all "
            "BM25 arms are fresh runtime-matched reruns. "
            "Labels above each pair report gains in percentage points over the "
            "paired BM25 arm. "
            "The pooled row contains 14,150 decisions across the five models for "
            "which both arms fit the context window."
        ),
        "alt_text": (
            "Grouped bar chart comparing end-to-end accuracy with fresh BM25 "
            "candidates and routed K=2 HySkill candidates for five eligible "
            "models and their pooled decisions."
        ),
    }


def main() -> None:
    """Generate the candidate-source ablation artwork and its evidence manifest."""
    configure_style()
    model_results = tuple(build_result(spec) for spec in MODEL_SPECS)
    instance_counts = {result["instance_count"] for result in model_results}
    if instance_counts != {HELD_OUT_PER_MODEL}:
        raise ValueError(
            f"Unexpected per-model instance counts: observed={tuple(sorted(instance_counts))}"
        )
    pooled_result = build_pooled_result(model_results)
    rows = build_plot_rows(model_results, pooled_result)
    color_figure = draw_chart(rows, COLOR_STYLES)
    grayscale_figure = draw_chart(rows, GRAYSCALE_STYLES)
    outputs = save_outputs(color_figure, grayscale_figure)
    plt.close(color_figure)
    plt.close(grayscale_figure)

    manifest = build_manifest(model_results, pooled_result, outputs)
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT_DIRECTORY),
                "pooled": pooled_result,
                "outputs": outputs,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
