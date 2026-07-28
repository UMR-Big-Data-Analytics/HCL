import os
from warnings import warn

import pandas as pd
import scienceplots  # noqa
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR


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
    df_mean = df[["gamma", "ari", "v_measure"]].groupby(["gamma"]).mean().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(5, 2), sharex=True)
    for ax, metric, ylabel in zip(axes, ["ari", "v_measure"], ["ARI", "V-Measure"], strict=True):
        for dataset_name in df["dataset_name"].unique():
            df_unique = df[df["dataset_name"] == dataset_name].sort_values("gamma")
            ax.plot(df_unique["gamma"], df_unique[metric], label=dataset_name, color="grey", alpha=0.5)
        ax.plot(df_mean.index, df_mean[metric], label="Mean", linestyle="--", color="blue")
        ax.set_xlabel("$\\gamma$")
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(ROOT_DIR / "results" / "plots" / "gamma.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
