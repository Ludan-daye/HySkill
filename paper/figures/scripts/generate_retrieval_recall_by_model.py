"""Generate per-model and available-model-average Recall@K figures."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from statistics import fmean
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from publication_figure_style import YAxisConfig, calculate_y_axis, configure_style


class ArchiveRetrievalRecord(TypedDict):
    """One row of the archived per-instance pack, which carries every variant."""

    instance_id: str
    domain: str
    variant: str
    gold: list[str]
    top50: list[tuple[str, float, int]]


class RetrievedItem(TypedDict):
    skill_id: str
    score: float


class K2RetrievalRecord(TypedDict):
    """One row of the active K=2 pack, which publishes routed retrieval only."""

    instance_id: str
    domain: str
    variant: str
    gold_skill_ids: list[str]
    retrieved: list[RetrievedItem]


class ModelSpec(TypedDict):
    slug: str
    label: str
    k2_path: Path
    k4_path: Path


class SeriesStyle(TypedDict):
    color: str
    linestyle: str
    marker: str


class ModelResult(TypedDict):
    spec: ModelSpec
    comparison_domains: tuple[str, ...]
    instance_count: int
    series: dict[str, tuple[float, ...]]
    omitted_methods: dict[str, str]


class ModelManifestEntry(TypedDict):
    label: str
    sources: dict[str, str]
    comparison_domains: tuple[str, ...]
    instance_count: int
    series: dict[str, tuple[float, ...]]
    omitted_methods: dict[str, str]
    y_axis: YAxisConfig
    pdf: str
    png: str


class AverageManifestEntry(TypedDict):
    label: str
    model_counts: dict[str, int]
    series: dict[str, tuple[float, ...]]
    y_axis: YAxisConfig
    pdf: str
    png: str


class Manifest(TypedDict):
    comparison: str
    metric: str
    k_provenance: dict[str, str]
    cutoffs: tuple[int, ...]
    domains: tuple[str, ...]
    style_mapping: dict[str, SeriesStyle]
    models: dict[str, ModelManifestEntry]
    average: AverageManifestEntry


RecordKey = tuple[str, str]
RankedRecord = tuple[tuple[str, ...], tuple[str, ...]]
MethodRecords = dict[str, dict[RecordKey, RankedRecord]]

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "paper" / "figures" / "retrieval-recall-by-model"
RESULT_TAGS: tuple[tuple[str, str], ...] = (
    ("01-qwen35-4b", "qwen3.5-4b-reference"),
    ("02-qwen35-9b", "qwen35-9b"),
    ("03-glm4-9b", "glm4-9b"),
    ("04-llama31-8b", "llama31-8b"),
    ("05-deepseek7b", "deepseek7b"),
    ("06-yi15-9b", "yi15-9b"),
    ("07-mistral7b", "mistral7b"),
)
MODEL_LABELS: dict[str, str] = {
    "01-qwen35-4b": "Qwen3.5-4B",
    "02-qwen35-9b": "Qwen3.5-9B",
    "03-glm4-9b": "GLM-4-9B",
    "04-llama31-8b": "Llama-3.1-8B",
    "05-deepseek7b": "DeepSeek-7B",
    "06-yi15-9b": "Yi-1.5-9B",
    "07-mistral7b": "Mistral-7B",
}


def model_spec(slug: str, result_tag: str) -> ModelSpec:
    """Bind one model to its K=2 pack and its K-independent baseline pack."""
    tag_root = REPOSITORY_ROOT / "community-results" / result_tag
    return {
        "slug": slug,
        "label": MODEL_LABELS[slug],
        "k2_path": tag_root / "k2" / "retrieval_top50.jsonl.gz",
        "k4_path": tag_root / "k4" / "retrieval_top50.jsonl.gz",
    }


MODEL_SPECS: tuple[ModelSpec, ...] = tuple(
    model_spec(slug, result_tag) for slug, result_tag in RESULT_TAGS
)
DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
    "bigcodebench",
)
MODEL_DOMAIN_POOLS: dict[str, tuple[str, ...]] = {
    "01-qwen35-4b": DOMAINS[:-1],
    "02-qwen35-9b": DOMAINS,
    "03-glm4-9b": DOMAINS,
    "04-llama31-8b": DOMAINS,
    "05-deepseek7b": DOMAINS,
    "06-yi15-9b": DOMAINS,
    "07-mistral7b": DOMAINS,
}
CUTOFFS: tuple[int, ...] = (5, 10, 50)
METHODS: tuple[str, ...] = (
    "routed",
    "llm_rerank",
    "bm25",
)
# Routed retrieval is imagination-dependent and must come from the active K=2
# pack. BM25 and the LLM reranker never read an imagination, so they are
# K-independent and are shared with the archived per-instance pack, which is the
# only place their top50 rankings are published.
K2_METHODS: frozenset[str] = frozenset({"routed"})
K_INDEPENDENT_METHODS: frozenset[str] = frozenset({"llm_rerank", "bm25"})
REFERENCE_METHOD = "routed"
STYLES: dict[str, SeriesStyle] = {
    "routed": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "llm_rerank": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    "bm25": {"color": "#666666", "linestyle": ":", "marker": "^"},
}


def store_record(
    method_records: MethodRecords,
    method: str,
    domain: str,
    instance_id: str,
    gold: tuple[str, ...],
    ranked_ids: tuple[str, ...],
    path: Path,
    line_number: int,
) -> None:
    """Validate and file one instance's ranking, refusing silent overwrites."""
    if not gold:
        raise ValueError(
            f"Empty gold set in {path} at line {line_number}: instance_id={instance_id}"
        )
    if len(ranked_ids) < max(CUTOFFS):
        raise ValueError(
            f"Short ranking in {path} at line {line_number}: "
            f"instance_id={instance_id}, length={len(ranked_ids)}"
        )
    key = (domain, instance_id)
    if key in method_records[method]:
        raise ValueError(
            f"Duplicate retrieval record in {path}: "
            f"method={method}, domain={domain}, instance_id={instance_id}"
        )
    method_records[method][key] = (gold, ranked_ids)


def read_jsonl(path: Path):
    """Yield parsed rows from a gzipped JSONL archive with line context."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing retrieval archive: {path}")
    with gzip.open(path, mode="rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error


def load_method_records(spec: ModelSpec) -> MethodRecords:
    """Load routed retrieval from the K=2 pack and the K-independent baselines.

    Routed rankings must come from the active K=2 pack. BM25 and the LLM reranker
    do not consume imaginations, so their rankings are identical under any K and
    are read from the archived per-instance pack that publishes them.
    """
    method_records: MethodRecords = {method: {} for method in METHODS}

    for line_number, raw in read_jsonl(spec["k2_path"]):
        record = cast(K2RetrievalRecord, raw)
        method = record["variant"]
        if method not in K2_METHODS:
            continue
        store_record(
            method_records,
            method,
            record["domain"],
            record["instance_id"],
            tuple(record["gold_skill_ids"]),
            tuple(item["skill_id"] for item in record["retrieved"]),
            spec["k2_path"],
            line_number,
        )

    for line_number, raw in read_jsonl(spec["k4_path"]):
        record = cast(ArchiveRetrievalRecord, raw)
        method = record["variant"]
        if method not in K_INDEPENDENT_METHODS:
            continue
        store_record(
            method_records,
            method,
            record["domain"],
            record["instance_id"],
            tuple(record["gold"]),
            tuple(candidate[0] for candidate in record["top50"]),
            spec["k4_path"],
            line_number,
        )

    return method_records


def recall_at_k(record: RankedRecord, cutoff: int) -> float:
    """Compute one instance's gold-skill recall at a retrieval cutoff."""
    gold, ranked_ids = record
    gold_ids = set(gold)
    retrieved_ids = set(ranked_ids[:cutoff])
    return len(gold_ids.intersection(retrieved_ids)) / len(gold_ids)


def build_model_result(spec: ModelSpec) -> ModelResult:
    """Build complete method series for one model without imputing missing data."""
    method_records = load_method_records(spec)
    comparison_domains = MODEL_DOMAIN_POOLS[spec["slug"]]
    comparison_domain_set = set(comparison_domains)
    reference_records = {
        key: record
        for key, record in method_records[REFERENCE_METHOD].items()
        if key[0] in comparison_domain_set
    }
    reference_keys = set(reference_records)
    reference_domains = {domain for domain, _ in reference_keys}
    if reference_domains != comparison_domain_set:
        raise ValueError(
            f"Reference method has an unexpected domain pool for {spec['label']}: "
            f"expected={comparison_domains}, observed={tuple(sorted(reference_domains))}"
        )

    series: dict[str, tuple[float, ...]] = {}
    omitted_methods: dict[str, str] = {}
    for method in METHODS:
        records = {
            key: record
            for key, record in method_records[method].items()
            if key[0] in comparison_domain_set
        }
        method_keys = set(records)
        if method_keys != reference_keys:
            omitted_methods[method] = (
                f"partial instance pool: {len(method_keys)}/{len(reference_keys)}"
            )
            continue
        ordered_records = tuple(records[key] for key in sorted(reference_keys))
        series[method] = tuple(
            fmean(recall_at_k(record, cutoff) for record in ordered_records)
            for cutoff in CUTOFFS
        )

    if not series:
        raise ValueError(f"No complete method series available for {spec['label']}")
    return {
        "spec": spec,
        "comparison_domains": comparison_domains,
        "instance_count": len(reference_keys),
        "series": series,
        "omitted_methods": omitted_methods,
    }


def build_average_series(
    model_results: tuple[ModelResult, ...],
) -> tuple[dict[str, tuple[float, ...]], dict[str, int]]:
    """Average each method over the models with a complete series for that method."""
    average_series: dict[str, tuple[float, ...]] = {}
    model_counts: dict[str, int] = {}
    for method in METHODS:
        available_series = tuple(
            result["series"][method]
            for result in model_results
            if method in result["series"]
        )
        if not available_series:
            raise ValueError(f"No complete model series available for method={method}")
        model_counts[method] = len(available_series)
        average_series[method] = tuple(
            fmean(series[index] for series in available_series)
            for index in range(len(CUTOFFS))
        )
    return average_series, model_counts


def draw_chart(
    series: dict[str, tuple[float, ...]], y_axis: YAxisConfig
) -> Figure:
    """Draw one Recall@K panel at its final four-column print size."""
    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=(1.66, 1.18), dpi=300)
    x_positions: tuple[int, ...] = tuple(range(len(CUTOFFS)))
    smooth_x = np.linspace(x_positions[0], x_positions[-1], 160)

    for method in METHODS:
        if method not in series:
            continue
        style = STYLES[method]
        coefficients = np.polyfit(x_positions, series[method], deg=2)
        smooth_y = np.polyval(coefficients, smooth_x)
        axes.plot(
            smooth_x,
            smooth_y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.45,
            zorder=2,
        )
        axes.plot(
            x_positions,
            series[method],
            color=style["color"],
            linestyle="none",
            marker=style["marker"],
            markersize=4.2,
            markeredgewidth=0.7,
            markerfacecolor="white",
            zorder=3,
        )

    axes.set_xticks(x_positions, [str(cutoff) for cutoff in CUTOFFS])
    axes.set_xticks((0.5, 1.5), minor=True)
    axes.set_xlim(-0.1, 2.1)
    axes.set_ylim(y_axis["limits"])
    axes.set_yticks(y_axis["ticks"])
    axes.grid(axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.8, zorder=0)
    axes.grid(
        axis="x",
        which="both",
        color="#D0D0D0",
        linewidth=0.55,
        alpha=0.8,
        zorder=0,
    )
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
    axes.tick_params(width=1.1, length=2.5)
    axes.tick_params(axis="x", which="minor", bottom=False, top=False)

    figure.subplots_adjust(left=0.23, right=0.98, bottom=0.18, top=0.97)
    return figure


def save_chart(figure: Figure, slug: str) -> tuple[Path, Path]:
    """Save one chart as vector artwork and a 300-DPI preview."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIRECTORY / f"{slug}.pdf"
    png_path = OUTPUT_DIRECTORY / f"{slug}.png"
    figure.savefig(pdf_path, format="pdf", facecolor="white")
    figure.savefig(png_path, format="png", dpi=300, facecolor="white")
    return pdf_path, png_path


def build_manifest(
    model_results: tuple[ModelResult, ...],
    model_files: dict[str, tuple[Path, Path]],
    model_y_axes: dict[str, YAxisConfig],
    average_series: dict[str, tuple[float, ...]],
    average_model_counts: dict[str, int],
    average_y_axis: YAxisConfig,
    average_files: tuple[Path, Path],
) -> Manifest:
    """Describe data availability and exact plotted values for reproducibility."""
    model_entries: dict[str, ModelManifestEntry] = {}
    for result in model_results:
        spec = result["spec"]
        pdf_path, png_path = model_files[spec["slug"]]
        model_entries[spec["slug"]] = {
            "label": spec["label"],
            "sources": {
                "routed_k2": str(spec["k2_path"].relative_to(REPOSITORY_ROOT)),
                "k_independent_baselines": str(
                    spec["k4_path"].relative_to(REPOSITORY_ROOT)
                ),
            },
            "comparison_domains": result["comparison_domains"],
            "instance_count": result["instance_count"],
            "series": result["series"],
            "omitted_methods": result["omitted_methods"],
            "y_axis": model_y_axes[spec["slug"]],
            "pdf": str(pdf_path.relative_to(REPOSITORY_ROOT)),
            "png": str(png_path.relative_to(REPOSITORY_ROOT)),
        }

    average_pdf, average_png = average_files
    return {
        "comparison": (
            "Gold-skill Recall@K for K=2 routed imagination, LLM reranking, "
            "and BM25"
        ),
        "k_provenance": {
            "routed": (
                "K_img=2 active pack (community-results/<tag>/k2/); "
                "imagination-dependent"
            ),
            "llm_rerank": (
                "K-independent: reads BM25 candidates, never an imagination; "
                "rankings identical under any K"
            ),
            "bm25": (
                "K-independent: lexical retrieval, never an imagination; "
                "rankings identical under any K"
            ),
        },
        "metric": (
            "Mean per-instance recall pooled over each figure's listed comparison "
            "domains; model figures do not average across models"
        ),
        "cutoffs": CUTOFFS,
        "domains": DOMAINS,
        "style_mapping": STYLES,
        "models": model_entries,
        "average": {
            "label": "Available-model average",
            "model_counts": average_model_counts,
            "series": average_series,
            "y_axis": average_y_axis,
            "pdf": str(average_pdf.relative_to(REPOSITORY_ROOT)),
            "png": str(average_png.relative_to(REPOSITORY_ROOT)),
        },
    }


def main() -> None:
    """Generate seven model figures, one available-model average, and a manifest."""
    configure_style()
    model_results = tuple(build_model_result(spec) for spec in MODEL_SPECS)
    model_files: dict[str, tuple[Path, Path]] = {}
    model_y_axes: dict[str, YAxisConfig] = {}

    for result in model_results:
        slug = result["spec"]["slug"]
        y_axis = calculate_y_axis(result["series"])
        figure = draw_chart(result["series"], y_axis)
        model_y_axes[slug] = y_axis
        model_files[slug] = save_chart(figure, slug)
        plt.close(figure)

    average_series, average_model_counts = build_average_series(model_results)
    average_y_axis = calculate_y_axis(average_series)
    average_figure = draw_chart(average_series, average_y_axis)
    average_files = save_chart(average_figure, "08-available-model-average")
    plt.close(average_figure)

    manifest = build_manifest(
        model_results,
        model_files,
        model_y_axes,
        average_series,
        average_model_counts,
        average_y_axis,
        average_files,
    )
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_directory": str(OUTPUT_DIRECTORY), **manifest["average"]}))


if __name__ == "__main__":
    main()
