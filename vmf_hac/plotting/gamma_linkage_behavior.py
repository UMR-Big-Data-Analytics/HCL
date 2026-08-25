import os

import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.plotting.consts import (
    DATASET_ORDER,
    categorical_palette,
    dataset_display_name,
    figure_size,
    save_figure,
    setup_publication_style,
)

K_MAX = 2500


def main():
    setup_publication_style()
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)

    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_gamma_linkage_behavior.csv")
    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    dataset_order = [label for label in DATASET_ORDER if label in set(df["dataset_label"])]
    gamma_order = sorted(df["gamma"].unique().tolist())
    gamma_palette = categorical_palette(gamma_order)

    # One panel per dataset stacked vertically: every curve gets the full column width,
    # which matters because k spans three decades on a log axis.
    fig, axes = plt.subplots(
        nrows=len(dataset_order),
        ncols=1,
        figsize=figure_size(columns=1, aspect=0.30 * len(dataset_order) + 0.25),
        sharex=True,
        sharey=False,
        layout="constrained",
    )

    for ax, dataset_label in zip(axes, dataset_order, strict=False):
        dataset_df = df[df["dataset_label"] == dataset_label].copy()
        n_true_clusters = int(dataset_df.iloc[0]["n_true_clusters"])
        sns.lineplot(
            data=dataset_df,
            x="k",
            y="ari_vmf",
            hue="gamma",
            hue_order=gamma_order,
            palette=gamma_palette,
            legend=False,
            linewidth=0.9,
            ax=ax,
        )
        max_val = dataset_df["ari_vmf"].max() * 1.05

        # ward_df = dataset_df[["k", "ari_ward"]].drop_duplicates().sort_values("k")
        # ax.plot(ward_df["k"], ward_df["ari_ward"], color="black", linestyle="--", linewidth=0.9)
        ax.axvline(float(n_true_clusters), color="red", linestyle=":", linewidth=0.9)

        ax.set_xscale("log")
        ax.set_xlim(left=1, right=K_MAX)
        ax.set_ylim(bottom=0.0, top=max_val)
        ax.set_xlabel("")
        ax.set_ylabel("ARI")
        ax.xaxis.set_minor_locator(LogLocator(subs=(2, 5)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        # An in-panel label is more compact than a title and keeps the rows close together.
        ax.text(
            0.985,
            0.93,
            dataset_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

    axes[-1].set_xlabel("$k$")
    # fig.supylabel("ARI")

    handles = [
        *(Line2D([0], [0], color=gamma_palette[gamma], label=f"$\\gamma = {gamma:g}$") for gamma in gamma_order),
        Line2D([0], [0], color="black", linestyle="--", label="Ward"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, ROOT_DIR / "results" / "plots" / "gamma_linkage_behavior.pdf")


if __name__ == "__main__":
    main()
