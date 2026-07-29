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
    fig, axes = plt.subplots(2, 2, figsize=(4, 3.5))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for ax, dataset_label in zip(axes, dataset_order, strict=False):
        dataset_df = df[df["dataset_name"] == dataset_label]
        n_true_clusters = int(dataset_df.iloc[0]["n_true_clusters"])
        linear_margin = max(2, round(0.3 * n_true_clusters))
        second_linear_start_k = n_true_clusters + linear_margin
        second_linear_scale = 0.35
        dataset_df = dataset_df.copy()
        dataset_df["k_plot"] = dataset_df["k"].map(
            lambda k: (
                k
                if k <= second_linear_start_k  # noqa: B023
                else second_linear_start_k + second_linear_scale * (k - second_linear_start_k)  # noqa: B023
            )
        )
        sns.lineplot(
            data=dataset_df,
            x="k_plot",
            y="ari_vmf",
            hue="gamma",
            hue_order=gamma_order,
            palette=gamma_palette,
            # marker="o",
            ax=ax,
            # markersize=4,
        )

        ward_df = dataset_df[["k_plot", "ari_ward"]].drop_duplicates().sort_values("k_plot")
        ax.plot(ward_df["k_plot"], ward_df["ari_ward"], color="black", linestyle="--", label="Ward")

        ax.axvline(n_true_clusters, color="red", linestyle="--", alpha=0.9)
        ax.axvline(second_linear_start_k, color="grey", linestyle="--", alpha=0.25)
        ax.set_title(dataset_label, fontsize=10)
        ax.set_xlabel("$k$")
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
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
        )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ROOT_DIR / "results" / "plots" / "gamma_linkage_behavior.pdf", bbox_inches="tight", dpi=300)
