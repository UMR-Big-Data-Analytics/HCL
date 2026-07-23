import os
from warnings import warn

import pandas as pd
import scienceplots  # noqa
import seaborn as sns
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
    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_dimensions.csv")

    n_cols = 4

    n_datasets = df["dataset_name"].unique().shape[0]
    rows, cols = (n_datasets // n_cols) + 1, n_cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=False)
    axes = axes.flatten()

    for i, (dataset, frame) in enumerate(df.groupby("dataset_name")):
        ax = axes[i]
        sns.lineplot(
            data=frame,
            x="reduction_factor",
            y="ari_mean",
            hue="method",
            marker="o",
            ax=ax,
        )
        ax.set_title(str(dataset))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    plt.savefig(ROOT_DIR / "results" / "plots" / "dimensions.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
