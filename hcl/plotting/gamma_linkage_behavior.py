import os
import pickle
import sys

import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.cluster.hierarchy import dendrogram

from hcl.core import HclHAC
from hcl.definitions import ROOT_DIR
from hcl.entity import DatasetManager, TextDatasets
from hcl.plotting.consts import (
    DATASET_ORDER,
    categorical_palette,
    dataset_display_name,
    figure_size,
    save_figure,
    setup_publication_style,
)

K_MAX = 2500


def linkage_trees():
    sys.setrecursionlimit(5000)
    cache_path = ROOT_DIR / "results" / "data" / "linkage_trees_dbpedia.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            linkage_small, linkage_large, linkage_progressive = pickle.load(cache_file)
    else:
        dataset_manager = DatasetManager("intfloat/multilingual-e5-large")
        dataset = dataset_manager.get(TextDatasets.DBPEDIA_14)
        k = np.unique(dataset.labels).shape[0]
        vmf_small = HclHAC(n_clusters=k, gamma=0.01)
        vmf_large = HclHAC(n_clusters=k, gamma=0.05)
        vmf_progressive = HclHAC(n_clusters=k, gamma=(0.01, 0.075), progressive=True)
        vmf_small.fit(dataset.embeddings)
        vmf_large.fit(dataset.embeddings)
        vmf_progressive.fit(dataset.embeddings)

        linkage_small = vmf_small.linkage_matrix_
        linkage_large = vmf_large.linkage_matrix_
        linkage_progressive = vmf_progressive.linkage_matrix_
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as cache_file:
            pickle.dump((linkage_small, linkage_large, linkage_progressive), cache_file)

    # plot the linkage tree
    fig, axes = plt.subplots(1, 3, figsize=figure_size(columns=1, aspect=0.5, scale=0.8), layout="constrained")
    for ax, gamma, linkage in zip(
        axes,
        [0.01, 0.05, (0.01, 0.075)],
        [linkage_small, linkage_large, linkage_progressive],
        strict=False,
    ):
        assert linkage is not None
        dendrogram(
            linkage,
            ax=ax,
            orientation="right",
            no_labels=True,
            color_threshold=np.inf,
            link_color_func=lambda _: "black",
            show_leaf_counts=False,
        )
        for collection in ax.collections:
            collection.set_linewidth(0.2)
        if isinstance(gamma, tuple):
            ax.set_title(r"$\gamma = 0.01$--$0.075$")
        else:
            ax.set_title(rf"$\gamma = {gamma}$")
        ax.set_xticks([])
    save_figure(fig, ROOT_DIR / "results" / "plots" / "linkage_tree_dbpedia.pdf")


def main():
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)

    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_gamma_linkage_behavior.csv")
    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    dataset_order = [label for label in DATASET_ORDER if label in set(df["dataset_label"])]
    gamma_order = sorted(df["gamma"].unique().tolist())
    gamma_palette = categorical_palette(gamma_order)
    gamma_order = [g for g in gamma_order if not str(g).startswith("(")]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figure_size(columns=1),
        sharex=True,
        sharey=False,
        layout="constrained",
    )
    axes = axes.ravel()

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

        df_vmf_prog = dataset_df[dataset_df["gamma"] == "(0.01, 0.075)"].drop_duplicates().sort_values("k")
        ax.plot(df_vmf_prog["k"], df_vmf_prog["ari_vmf"], color=gamma_palette["(0.01, 0.075)"], linestyle="--")
        ax.axvline(float(n_true_clusters), color="red", linestyle=":", linewidth=0.9)

        ax.set_xscale("log")
        ax.set_xlim(left=1, right=K_MAX)
        ax.set_ylim(bottom=0.0, top=max_val)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(LogLocator(subs=(2, 5), numticks=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(dataset_label, pad=2)

    for ax in axes[2:]:
        ax.set_xlabel("$k$")
    # Tick labels differ in width (0.25 vs 0.5)
    for ax in axes[::2]:
        ax.set_ylabel("ARI")
        ax.yaxis.set_label_coords(-0.22, 0.5)

    handles = [
        *(Line2D([0], [0], color=gamma_palette[gamma], label=f"$\\gamma = {gamma}$") for gamma in gamma_order),
        Line2D(
            [0],
            [0],
            color=gamma_palette["(0.01, 0.075)"],
            linestyle="--",
            label=r"$\gamma = 0.01$--$0.075$",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, ROOT_DIR / "results" / "plots" / "gamma_linkage_behavior.pdf")


if __name__ == "__main__":
    setup_publication_style()
    linkage_trees()
    main()
