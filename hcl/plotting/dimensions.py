import os
from math import ceil

import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt

from hcl.definitions import ROOT_DIR
from hcl.plotting.consts import (
    CLUSTERER_LABELS,
    categorical_palette,
    clusterer_label_order,
    dataset_display_name,
    figure_size,
    page_aspect_limit,
    save_figure,
    setup_publication_style,
)
from hcl.plotting.latex_table import style_top3_latex


def _compute_v_measure_auc_table(df: pd.DataFrame, metric: str = "v_measure_effective") -> pd.DataFrame:
    auc_rows: list[dict[str, object]] = []
    for (clusterer_label, dataset_label), frame in df.groupby(["clusterer_label", "dataset_label"]):
        curve = frame.groupby("reduction_factor")[metric].mean().reset_index().sort_values("reduction_factor")
        auc_rows.append(
            {
                "clusterer_label": clusterer_label,
                "dataset_label": dataset_label,
                "auc": float(np.trapezoid(curve[metric].to_numpy(), curve["reduction_factor"].to_numpy())),
            }
        )

    auc_df = pd.DataFrame(auc_rows)
    table = auc_df.pivot(index="dataset_label", columns="clusterer_label", values="auc")
    algorithm_columns = clusterer_label_order(set(table.columns))
    table = table.reindex(columns=algorithm_columns)

    table.loc["Mean"] = table[algorithm_columns].mean(axis=0)
    return table


def main():
    setup_publication_style()
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)
    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_dimensions.csv")
    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    df["v_measure_effective"] = df["v_measure"] * (1 - df["noise_fraction"])
    df = (
        df[
            [
                "clusterer_name",
                "dataset_label",
                "dim",
                "ari",
                "ari_nearest",
                "v_measure",
                "v_measure_effective",
                "v_measure_nearest",
                "nmi",
                "ami",
                "ami_nearest",
                "noise_fraction",
                "homogeneity",
                "completeness",
                "fowlkes_mallows",
                "calinski_harabasz",
                "davies_bouldin",
                "reduction_factor",
                "running_time_s",
            ]
        ]
        .groupby(["clusterer_name", "dataset_label", "dim"])
        .mean()
        .reset_index()
    )
    df["clusterer_label"] = df["clusterer_name"].map(CLUSTERER_LABELS).fillna(df["clusterer_name"])
    label_order = clusterer_label_order(set(df["clusterer_label"]))
    clusterer_palette = categorical_palette(label_order)

    n_datasets = df["dataset_label"].unique().shape[0]
    cols = min(2, n_datasets)
    rows = ceil(n_datasets / cols)
    aspect = min(rows / cols, page_aspect_limit(columns=1))
    fig = plt.figure(figsize=figure_size(columns=1, aspect=aspect), layout="constrained")
    axes = fig.subplots(rows, cols, sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)
    legend_handles = None
    legend_labels = None

    for i, (dataset_label, frame) in enumerate(df.groupby("dataset_label")):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        sns.lineplot(
            data=frame,
            x="reduction_factor",
            y="ami_nearest",
            hue="clusterer_label",
            hue_order=label_order,
            palette=clusterer_palette,
            ax=ax,
        )
        ax.set_title(dataset_label)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("")
        ax.set_ylabel("")
        handles, labels = ax.get_legend_handles_labels()
        if legend_handles is None and labels:
            legend_handles = handles
            legend_labels = labels
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    n_plotted = i + 1
    for j in range(n_plotted, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].set_visible(False)

    # sharex hides the tick labels of every axis but the last row, which can be partly empty.
    for col in range(cols):
        last_row = max(row for row in range(rows) if row * cols + col < n_plotted)
        ax = axes[last_row, col]
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("$\\alpha$")
        ax.xaxis.label.set_visible(True)

    fig.supylabel("AMI (noise reassigned)")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="outside lower center",
            ncol=min(len(legend_labels), 3),
            frameon=False,
        )

    save_figure(fig, ROOT_DIR / "results" / "plots" / "dimensions.pdf")

    os.makedirs(ROOT_DIR / "results" / "tables", exist_ok=True)
    auc_table_latex = style_top3_latex(_compute_v_measure_auc_table(df, metric="v_measure_effective"))
    with open(
        ROOT_DIR / "results" / "tables" / "dimensions_v_measure_effective_auc.tex", "w", encoding="utf-8"
    ) as file:
        file.write(auc_table_latex)


if __name__ == "__main__":
    main()
