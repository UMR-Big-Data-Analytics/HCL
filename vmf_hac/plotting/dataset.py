import os

import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from critdd import Diagram, Diagrams
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity.dataset import DatasetFactory
from vmf_hac.plotting.consts import (
    categorical_palette,
    figure_size,
    page_aspect_limit,
    save_figure,
    setup_publication_style,
)
from vmf_hac.plotting.embedding_models import CLUSTERER_LABELS

METRICS = {
    "v_measure": ("V-Measure", "datasets_v_measure.pdf"),
    "ari": ("ARI", "datasets_ari.pdf"),
}
LANDSCAPE_ASPECT = 1 / np.sqrt(2)


def _clusterer_order(df: pd.DataFrame) -> list[str]:
    order = [CLUSTERER_LABELS[name] for name in CLUSTERER_LABELS if name in set(df["clusterer_name"])]
    order.extend(sorted(set(df["clusterer_label"]) - set(order)))
    return order


def _plot_metric_table(
    df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    filename: str,
) -> None:
    dataset_order = sorted(df["dataset_label"].unique())
    clusterer_order = _clusterer_order(df)
    clusterer_palette = categorical_palette(clusterer_order)

    rows = len(dataset_order)
    cols = len(clusterer_order)
    aspect = min(0.55 * rows / cols, LANDSCAPE_ASPECT, page_aspect_limit(columns=2))
    fig = plt.figure(figsize=figure_size(columns=2, aspect=aspect), layout="constrained")
    axes = np.atleast_2d(fig.subplots(rows, cols, sharex=True)).reshape(rows, cols)

    metric_min = float(df[metric].min())
    lower_limit = min(0.0, metric_min)
    padding = 0.02

    for row, dataset_label in enumerate(dataset_order):
        dataset_df = df[df["dataset_label"] == dataset_label]
        for col, clusterer_label in enumerate(clusterer_order):
            ax = axes[row, col]
            panel_df = dataset_df[dataset_df["clusterer_label"] == clusterer_label]
            if panel_df.empty:
                ax.text(0.5, 0.5, "--", ha="center", va="center", transform=ax.transAxes)
            else:
                sns.violinplot(
                    data=panel_df,
                    x=metric,
                    color=clusterer_palette[clusterer_label],
                    cut=0,
                    inner="quart",
                    linewidth=0.5,
                    width=0.85,
                    ax=ax,
                )

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.set_xlim(lower_limit - padding, 1.0 + padding)
            ax.grid(axis="x", alpha=0.2, linewidth=0.4)
            ax.tick_params(axis="x", labelsize=5, pad=1)
            if row == 0:
                ax.set_title(clusterer_label, fontsize=6)
            if col == 0:
                ax.set_ylabel(dataset_label, rotation=0, ha="right", va="center", fontsize=6)

    fig.supxlabel(metric_label)
    save_figure(fig, ROOT_DIR / "results" / "plots" / filename)
    plt.close(fig)


def _write_critical_difference_diagram(
    df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
) -> None:
    scores = _metric_scores(df, metric=metric, metric_label=metric_label)
    diagram = Diagram(
        scores.to_numpy(),
        treatment_names=scores.columns,
        maximize_outcome=True,
    )
    options = {
        "alpha": 0.05,
        "adjustment": "holm",
        "reverse_x": True,
        "axis_options": {"title": metric_label},
    }
    stem = f"datasets_{metric}_cd"
    diagram.to_file(ROOT_DIR / "results" / "tables" / f"{stem}.tex", **options)
    diagram.to_file(ROOT_DIR / "results" / "plots" / f"{stem}.pdf", **options)


def _metric_scores(
    df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
) -> pd.DataFrame:
    scores = df.pivot_table(
        index="dataset_label",
        columns="clusterer_label",
        values=metric,
        aggfunc="mean",
    ).reindex(columns=_clusterer_order(df))
    if scores.isna().any(axis=None):
        missing = int(scores.isna().sum().sum())
        msg = f"Cannot create {metric_label} critical difference diagram: {missing} scores are missing"
        raise ValueError(msg)
    return scores


def _write_2d_critical_difference_diagram(
    df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
) -> None:
    dataset_order = sorted(df["dataset_label"].unique())
    clusterer_order = _clusterer_order(df)
    score_tables = []
    for dataset_label in dataset_order:
        scores = (
            df[df["dataset_label"] == dataset_label]
            .pivot(
                index="split",
                columns="clusterer_label",
                values=metric,
            )
            .reindex(columns=clusterer_order)
        )
        if scores.isna().any(axis=None):
            missing = int(scores.isna().sum().sum())
            msg = f"Cannot create {metric_label} 2D diagram for {dataset_label}: {missing} scores are missing"
            raise ValueError(msg)
        score_tables.append(scores)

    diagram = Diagrams(
        np.stack([scores.to_numpy() for scores in score_tables]),
        diagram_names=dataset_order,
        treatment_names=clusterer_order,
        maximize_outcome=True,
    )
    options = {
        "alpha": 0.05,
        "adjustment": "holm",
        "axis_options": {
            "width": r"\axisdefaultwidth",
            "height": r"2.2*\axisdefaultheight",
            "title": metric_label,
        },
    }
    stem = f"datasets_{metric}_cd_2d"
    diagram.to_file(ROOT_DIR / "results" / "tables" / f"{stem}.tex", **options)
    diagram.to_file(ROOT_DIR / "results" / "plots" / f"{stem}.pdf", **options)


def main() -> None:
    setup_publication_style()
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)
    os.makedirs(ROOT_DIR / "results" / "tables", exist_ok=True)

    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_datasets.csv")
    fallback_labels = df["dataset_name"].str.rsplit("/").str[-1]
    df["dataset_label"] = df["dataset_name"].map(DatasetFactory.get_dataset_name_map()).fillna(fallback_labels)
    df["clusterer_label"] = df["clusterer_name"].map(CLUSTERER_LABELS).fillna(df["clusterer_name"])

    for metric, (metric_label, filename) in METRICS.items():
        _plot_metric_table(
            df,
            metric=metric,
            metric_label=metric_label,
            filename=filename,
        )
        _write_critical_difference_diagram(
            df,
            metric=metric,
            metric_label=metric_label,
        )
        _write_2d_critical_difference_diagram(
            df,
            metric=metric,
            metric_label=metric_label,
        )


if __name__ == "__main__":
    main()
