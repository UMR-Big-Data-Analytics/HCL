import os

import pandas as pd
import scienceplots  # noqa
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.plotting.consts import dataset_display_name, figure_size, save_figure, setup_publication_style


def main():
    setup_publication_style()
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)

    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_gamma.csv")
    df["gamma"] = df["params"].apply(lambda p: eval(p)["gamma"])

    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    df_mean = df[["gamma", "ari", "v_measure"]].groupby(["gamma"]).mean().sort_index()
    dataset_labels = sorted(df["dataset_label"].unique().tolist())

    fig, axes = plt.subplots(2, 1, figsize=figure_size(columns=1, aspect=1.0), sharex=True, layout="constrained")
    for ax, metric, ylabel in zip(axes, ["ari", "v_measure"], ["ARI", "V-Measure"], strict=True):
        for dataset_label in dataset_labels:
            df_unique = df[df["dataset_label"] == dataset_label].sort_values("gamma")
            ax.plot(df_unique["gamma"], df_unique[metric], color="grey", alpha=0.5)
        ax.plot(df_mean.index, df_mean[metric], linestyle="--", color="blue")
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("$\\gamma$")

    save_figure(fig, ROOT_DIR / "results" / "plots" / "gamma.pdf")


if __name__ == "__main__":
    main()
