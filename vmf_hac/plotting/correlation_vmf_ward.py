from warnings import warn

import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity.dataset import TextDatasets
from vmf_hac.plotting.consts import DATASET_NAME_MAP, DATASET_ORDER, DATASET_PALETTE


def main():
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
            }
        )
    except:  # noqa
        warn("Warning: LaTeX not available. Using default matplotlib text rendering.", stacklevel=2)
    n_markers = 20
    datasets = [
        TextDatasets.BANKING77,
        TextDatasets.DBPEDIA_14,
        TextDatasets.TWENTY_NEWSGROUPS,
        TextDatasets.WIKICITIES,
    ]
    dataset_names = [dataset.value.__str__() for dataset in datasets]
    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_vmf_ward.csv")
    df["dataset_name"] = df["dataset_name"].astype(str)
    df = df[df["dataset_name"].isin(dataset_names)]
    df_mean = df.groupby(["dataset_name", "gamma"]).mean().reset_index()
    gammas = sorted(df_mean["gamma"].unique().tolist())
    new_gammas = []
    for i in range(n_markers):
        new_gammas.append(gammas[int(i * len(gammas) / n_markers)])
    df_mean = df_mean[df_mean["gamma"].isin(new_gammas)]

    plt.style.use(["science"])

    plot_df = df_mean[df_mean["gamma"] <= 1].copy()

    plot_df["dataset_label"] = plot_df["dataset_name"].map(DATASET_NAME_MAP)
    labels = [label for label in DATASET_ORDER if label in set(plot_df["dataset_label"].dropna().unique())]
    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "8"]
    marker_map = {lab: marker_cycle[i % len(marker_cycle)] for i, lab in enumerate(labels)}
    palette = {label: DATASET_PALETTE[label] for label in labels}

    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=plot_df,
        x="gamma",
        y="pearson_corr",
        hue="dataset_label",
        hue_order=labels,
        style="dataset_label",
        markers=marker_map,
        palette=palette,
        s=45,
        edgecolor="none",
        ax=ax,
    )

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Correlation")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, -0.15),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(ROOT_DIR / "results" / "plots" / "pearson_corr_gamma.pdf", bbox_inches="tight")

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.lineplot(
        data=plot_df,
        x="gamma",
        y="vmf_cluster_skew",
        hue="dataset_label",
        hue_order=labels,
        markers=marker_map,
        palette=palette,
        style="dataset_label",
        ax=ax,
    )
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Skewness")
    ward_skew_per_dataset = plot_df.groupby("dataset_label")["ward_cluster_skew"].mean()
    for dataset_label, ward_skew in ward_skew_per_dataset.items():
        ax.axhline(y=ward_skew, color=palette[dataset_label], linestyle="--", linewidth=1)  # ty:ignore[invalid-argument-type]
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT_DIR / "results" / "plots" / "skewness_gamma.pdf", bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
