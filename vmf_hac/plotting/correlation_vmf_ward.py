import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity.dataset import TextDatasets
from vmf_hac.plotting.consts import (
    DATASET_ORDER,
    DATASET_PALETTE,
    dataset_display_name,
    figure_size,
    save_figure,
    setup_publication_style,
)


def main():
    setup_publication_style()
    n_markers = 20
    datasets = [
        TextDatasets.BUILT_BENCH_CLUSTERING_P2P,
        TextDatasets.DBPEDIA_14,
        TextDatasets.CLUSTREC_COVID,
        TextDatasets.WIKICITIES,
    ]
    dataset_names = [dataset.value.technical_name for dataset in datasets]
    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_vmf_ward.csv")
    df["dataset_name"] = df["dataset_name"].astype(str)
    df = df[df["dataset_name"].isin(dataset_names)]
    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    df_mean = df.groupby(["dataset_label", "gamma"], as_index=False).mean(numeric_only=True)
    gammas = sorted(df_mean["gamma"].unique().tolist())
    new_gammas = []
    for i in range(n_markers):
        new_gammas.append(gammas[int(i * len(gammas) / n_markers)])
    df_mean = df_mean[df_mean["gamma"].isin(new_gammas)]

    plot_df = df_mean[df_mean["gamma"] <= 1].copy()

    labels = [label for label in DATASET_ORDER if label in set(plot_df["dataset_label"].dropna().unique())]
    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "8"]
    marker_map = {lab: marker_cycle[i % len(marker_cycle)] for i, lab in enumerate(labels)}
    palette = {label: DATASET_PALETTE[label] for label in labels}

    fig, ax = plt.subplots(figsize=figure_size(columns=1, aspect=0.8), layout="constrained")

    sns.scatterplot(
        data=plot_df,
        x="gamma",
        y="pearson_corr",
        hue="dataset_label",
        hue_order=labels,
        style="dataset_label",
        markers=marker_map,
        palette=palette,
        s=18,
        edgecolor="none",
        ax=ax,
    )

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Cophenetic correlation")
    handles, legend_labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()  # ty:ignore[unresolved-attribute]
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    save_figure(fig, ROOT_DIR / "results" / "plots" / "pearson_corr_gamma.pdf")

    plt.close(fig)

    fig, ax = plt.subplots(figsize=figure_size(columns=1, aspect=0.8), layout="constrained")
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
    handles, legend_labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()  # ty:ignore[unresolved-attribute]
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    save_figure(fig, ROOT_DIR / "results" / "plots" / "skewness_gamma.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
