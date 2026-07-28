import os
from warnings import warn

import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.plotting.consts import DATASET_ORDER, categorical_palette

if __name__ == "__main__":
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

    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_gamma_linkage_behavior.csv")
    dataset_order = DATASET_ORDER
    gamma_order = sorted(df["gamma"].unique().tolist())
    gamma_palette = categorical_palette(gamma_order)
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for ax, dataset_label in zip(axes, dataset_order, strict=False):
        dataset_df = df[df["dataset_name"] == dataset_label]
        sns.lineplot(
            data=dataset_df,
            x="k",
            y="ari_vmf",
            hue="gamma",
            hue_order=gamma_order,
            palette=gamma_palette,
            marker="o",
            ax=ax,
            markersize=4,
        )

        ward_df = dataset_df[["k", "ari_ward"]].drop_duplicates().sort_values("k")
        ax.plot(ward_df["k"], ward_df["ari_ward"], color="black", linestyle="--", label="Ward")

        n_true_clusters = int(dataset_df.iloc[0]["n_true_clusters"])
        ax.axvline(n_true_clusters, color="red", linestyle="--")
        ax.set_title(dataset_label)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("ARI")
        handles, labels = ax.get_legend_handles_labels()
        if legend_handles is None and labels:
            legend_handles = handles
            legend_labels = labels
        if ax.get_legend() is not None:
            ax.get_legend().remove()

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
    fig.savefig(ROOT_DIR / "results" / "plots" / "gamma_linkage_behavior.pdf", bbox_inches="tight", dpi=300)
