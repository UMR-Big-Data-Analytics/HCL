from warnings import warn

import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity.dataset import TextDatasets


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

    dataset_name_map = {
        "mteb/WikiCitiesClustering": "WikiCities",
        "mteb/llm-eval-banking77": "Banking77",
        "mteb/llm-eval-dbpedia_14": "DBPedia",
        "sklearn/20newsgroups": "20Newsgroups",
    }
    plot_df["dataset_label"] = plot_df["dataset_name"].map(dataset_name_map)

    labels = list(plot_df["dataset_label"].unique())
    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "8"]
    marker_map = {lab: marker_cycle[i % len(marker_cycle)] for i, lab in enumerate(labels)}
    palette = dict(zip(labels, sns.color_palette("colorblind", n_colors=len(labels)), strict=False))

    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=plot_df,
        x="gamma",
        y="pearson_corr",
        hue="dataset_label",
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


if __name__ == "__main__":
    main()
