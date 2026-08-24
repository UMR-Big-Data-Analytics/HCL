import os
from math import ceil

import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.plotting.consts import (
    categorical_palette,
    dataset_display_name,
    figure_size,
    page_aspect_limit,
    save_figure,
    setup_publication_style,
)

CLUSTERER_LABELS = {
    "AgglomerativeClustering_average": "Average",
    "AgglomerativeClustering_complete": "Complete",
    "AgglomerativeClustering_single": "Single",
    "AgglomerativeClustering_ward": "Ward",
    "KMeans": "K-Means",
    "SpectralClustering": "Spectral",
    "SphericalKMeans": "Spherical KM",
    "VmfHAC": "vMF-HAC",
    "VonMisesFisherMixture_soft": "moVMF",
}


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _compute_v_measure_auc_table(df: pd.DataFrame) -> pd.DataFrame:
    auc_rows: list[dict[str, object]] = []
    for (clusterer_label, dataset_label), frame in df.groupby(["clusterer_label", "dataset_label"]):
        curve = frame.groupby("reduction_factor")["v_measure"].mean().reset_index().sort_values("reduction_factor")
        auc_rows.append(
            {
                "clusterer_label": clusterer_label,
                "dataset_label": dataset_label,
                "auc": float(np.trapezoid(curve["v_measure"].to_numpy(), curve["reduction_factor"].to_numpy())),
            }
        )

    auc_df = pd.DataFrame(auc_rows)
    table = auc_df.pivot(index="dataset_label", columns="clusterer_label", values="auc")
    algorithm_columns = [label for label in CLUSTERER_LABELS.values() if label in set(table.columns)]
    table = table.reindex(columns=algorithm_columns)

    mean_algorithms = table[algorithm_columns].mean(axis=0)
    mean_row = {column: mean_algorithms[column] for column in algorithm_columns}
    table.loc["mean"] = pd.Series(mean_row)
    return table


def _style_top3_latex(table: pd.DataFrame) -> str:
    algorithm_columns = list(table.columns)

    formatted = table.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:.3f}")

    for idx in formatted.index:
        row_values = table.loc[idx, algorithm_columns]
        row_ranks = row_values.rank(ascending=False, method="min")
        for col in algorithm_columns:
            rank = row_ranks.loc[col]
            value = formatted.loc[idx, col]
            if rank == 1:
                formatted.loc[idx, col] = f"\\textbf{{\\uline{{{value}}}}}"
            elif rank == 2:
                formatted.loc[idx, col] = f"\\textbf{{\\dashuline{{{value}}}}}"
            elif rank == 3:
                formatted.loc[idx, col] = f"\\textbf{{\\dotuline{{{value}}}}}"

    escaped_columns = [_latex_escape(str(col)) for col in formatted.columns]
    formatted.columns = escaped_columns
    formatted.index = [_latex_escape(str(idx)) for idx in formatted.index]
    if "mean" in formatted.index:
        formatted.rename(index={"mean": r"\textbf{Mean}"}, inplace=True)
    formatted.index.name = "Dataset"
    latex = formatted.to_latex(escape=False, float_format=None)
    return latex.replace("\n\\textbf{Mean} &", "\n\\midrule\n\\textbf{Mean} &", 1)


def main():
    setup_publication_style()
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)
    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_dimensions.csv")
    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    df = (
        df[
            [
                "clusterer_name",
                "dataset_label",
                "dim",
                "ari",
                "v_measure",
                "nmi",
                "ami",
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
    clusterer_order = [name for name in CLUSTERER_LABELS if name in set(df["clusterer_name"].unique())]
    label_order = [CLUSTERER_LABELS[name] for name in clusterer_order]
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
            y="v_measure",
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

    fig.supylabel("V-Measure")

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
    auc_table = _compute_v_measure_auc_table(df)
    auc_table_latex = _style_top3_latex(auc_table)
    with open(ROOT_DIR / "results" / "tables" / "dimensions_v_measure_auc.tex", "w", encoding="utf-8") as file:
        file.write(auc_table_latex)


if __name__ == "__main__":
    main()
