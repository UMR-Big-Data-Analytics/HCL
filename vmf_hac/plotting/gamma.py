import os
from warnings import warn

import pandas as pd
import scienceplots  # noqa
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.plotting.consts import DATASET_NAME_MAP, DATASET_PALETTE, categorical_palette


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

    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_gamma.csv")
    df["gamma"] = df["params"].apply(lambda p: eval(p)["gamma"])
    df["dataset_label"] = df["dataset_name"].map(DATASET_NAME_MAP).fillna(df["dataset_name"])
    df_mean = df[["gamma", "ari", "v_measure"]].groupby(["gamma"]).mean().sort_index()
    dataset_labels = sorted(df["dataset_label"].unique().tolist())
    dynamic_palette = categorical_palette(dataset_labels)
    palette = {label: DATASET_PALETTE.get(label, dynamic_palette[label]) for label in dataset_labels}

    fig, axes = plt.subplots(1, 2, figsize=(5, 2), sharex=True)
    for ax, metric, ylabel in zip(axes, ["ari", "v_measure"], ["ARI", "V-Measure"], strict=True):
        for dataset_label in dataset_labels:
            df_unique = df[df["dataset_label"] == dataset_label].sort_values("gamma")
            ax.plot(df_unique["gamma"], df_unique[metric], label=dataset_label, color=palette[dataset_label], alpha=0.5)
        ax.plot(df_mean.index, df_mean[metric], label="Mean", linestyle="--", color="black")
        ax.set_xlabel("$\\gamma$")
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(ROOT_DIR / "results" / "plots" / "gamma.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
