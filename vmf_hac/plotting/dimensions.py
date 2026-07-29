import os
from warnings import warn

import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.plotting.consts import categorical_palette

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
    for (clusterer_label, dataset_name), frame in df.groupby(["clusterer_label", "dataset_name"]):
        curve = frame.groupby("reduction_factor")["v_measure"].mean().reset_index().sort_values("reduction_factor")
        auc_rows.append(
            {
                "clusterer_label": clusterer_label,
                "dataset_name": dataset_name,
                "auc": float(np.trapezoid(curve["v_measure"].to_numpy(), curve["reduction_factor"].to_numpy())),
            }
        )

    auc_df = pd.DataFrame(auc_rows)
    table = auc_df.pivot(index="dataset_name", columns="clusterer_label", values="auc")
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
                formatted.loc[idx, col] = f"\\textbf{{{value}}}"
            elif rank == 2:
                formatted.loc[idx, col] = f"\\underline{{{value}}}"
            elif rank == 3:
                formatted.loc[idx, col] = f"\\texttt{{{value}}}"

    escaped_columns = [_latex_escape(str(col)) for col in formatted.columns]
    formatted.columns = escaped_columns
    formatted.index = [_latex_escape(str(idx)) for idx in formatted.index]
    if "mean" in formatted.index:
        formatted.rename(index={"mean": r"\textbf{Mean}"}, inplace=True)
    formatted.index.name = "Dataset"
    latex = formatted.to_latex(escape=False, float_format=None)
    return latex.replace("\n\\textbf{Mean} &", "\n\\midrule\n\\textbf{Mean} &", 1)


def main():
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
            }
        )
    except:  # noqa
        warn("Warning: LaTeX not available. Using default matplotlib text rendering.", stacklevel=2)

    plt.style.use("science")
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)
    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_dimensions.csv")
    df = (
        df[
            [
                "clusterer_name",
                "dataset_name",
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
        .groupby(["clusterer_name", "dataset_name", "dim"])
        .mean()
        .reset_index()
    )
    df["clusterer_label"] = df["clusterer_name"].map(CLUSTERER_LABELS).fillna(df["clusterer_name"])
    clusterer_order = [name for name in CLUSTERER_LABELS if name in set(df["clusterer_name"].unique())]
    label_order = [CLUSTERER_LABELS[name] for name in clusterer_order]
    clusterer_palette = categorical_palette(label_order)

    n_cols = 5

    n_datasets = df["dataset_name"].unique().shape[0]
    rows, cols = (n_datasets // n_cols) + 1, n_cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows), sharey=False)
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for i, (dataset, frame) in enumerate(df.groupby("dataset_name")):
        ax = axes[i]
        sns.lineplot(
            data=frame,
            x="reduction_factor",
            y="v_measure",
            hue="clusterer_label",
            hue_order=label_order,
            palette=clusterer_palette,
            # marker="o",
            ax=ax,
        )
        ax.set_title(str(dataset), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("V-Measure")
        ax.set_xlabel("$\\alpha$")
        handles, labels = ax.get_legend_handles_labels()
        if legend_handles is None and labels:
            legend_handles = handles
            legend_labels = labels
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=3,
            frameon=False,
        )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(ROOT_DIR / "results" / "plots" / "dimensions.pdf", bbox_inches="tight", dpi=300)

    os.makedirs(ROOT_DIR / "results" / "tables", exist_ok=True)
    auc_table = _compute_v_measure_auc_table(df)
    auc_table_latex = _style_top3_latex(auc_table)
    with open(ROOT_DIR / "results" / "tables" / "dimensions_v_measure_auc.tex", "w", encoding="utf-8") as file:
        file.write(auc_table_latex)


if __name__ == "__main__":
    main()
