import os

import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.plotting.consts import (
    DATASET_ORDER,
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

MODEL_LABELS = {
    "intfloat/multilingual-e5-large": "mE5-large",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-Emb-8B",
    "codefuse-ai/F2LLM-v2-14B": "F2LLM-v2-14B",
}


def _plot_grid(df: pd.DataFrame, *, normalize: bool, filename: str) -> None:
    clusterer_order = [
        CLUSTERER_LABELS[name] for name in CLUSTERER_LABELS if name in set(df["clusterer_name"].unique())
    ]
    clusterer_palette = categorical_palette(clusterer_order)
    model_order = [MODEL_LABELS[name] for name in MODEL_LABELS if name in set(df["model_name"].unique())]
    dataset_order = [label for label in DATASET_ORDER if label in set(df["dataset_label"])]

    # One column per embedding model, one row per dataset: the k grid differs between
    # datasets, so their curves cannot share a panel.
    cols = len(model_order)
    rows = len(dataset_order)
    aspect = min(0.75 * rows / cols, page_aspect_limit(columns=2))
    fig = plt.figure(figsize=figure_size(columns=2, aspect=aspect), layout="constrained")
    axes = np.atleast_2d(fig.subplots(rows, cols, sharex="row", sharey="row")).reshape(rows, cols)

    legend_handles: list = []
    legend_labels: list = []

    for row, dataset_label in enumerate(dataset_order):
        dataset_df = df[df["dataset_label"] == dataset_label]
        n_true_clusters = int(dataset_df.iloc[0]["n_true_clusters"])
        for col, model_label in enumerate(model_order):
            ax = axes[row, col]
            panel_df = dataset_df[dataset_df["model_label"] == model_label].sort_values("n_clusters").copy()
            if normalize:
                # Dividing by the best ARI reached in the panel removes the level shift the
                # embedding model induces, so only the relative ordering remains visible.
                panel_df["v_measure"] = panel_df["v_measure"] / panel_df["v_measure"].max()
            sns.lineplot(
                data=panel_df,
                x="n_clusters",
                y="v_measure",
                hue="clusterer_label",
                hue_order=clusterer_order,
                palette=clusterer_palette,
                linewidth=0.9,
                ax=ax,
            )
            ax.axvline(float(n_true_clusters), color="red", linestyle=":", linewidth=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if row == 0:
                ax.set_title(model_label)
            if col == 0:
                ax.set_ylabel(dataset_label)

            handles, labels = ax.get_legend_handles_labels()
            if not legend_handles and labels:
                legend_handles, legend_labels = handles, labels
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    for ax in axes[-1, :]:
        ax.set_xlabel("$k$")

    fig.supylabel("V-measure / panel max" if normalize else "V-measure")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="outside lower center",
            ncol=min(len(legend_labels), 5),
            frameon=False,
        )

    save_figure(fig, ROOT_DIR / "results" / "plots" / filename)


def main():
    setup_publication_style()
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)

    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_embedding_models.csv")
    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    df["clusterer_label"] = df["clusterer_name"].map(CLUSTERER_LABELS).fillna(df["clusterer_name"])
    df["model_label"] = df["model_name"].map(MODEL_LABELS).fillna(df["model_name"])

    _plot_grid(df, normalize=False, filename="embedding_models.pdf")
    _plot_grid(df, normalize=True, filename="embedding_models_normalized.pdf")


if __name__ == "__main__":
    main()
