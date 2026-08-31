"""Rank-based views of the embedding-model sweep.

The per-panel ARI curves make it hard to see whether the *ordering* of the algorithms
survives a change of the embedding model, because the models also shift the overall
level of ARI. These figures separate the two effects.
"""

import os

import numpy as np
import pandas as pd
import scienceplots  # noqa
from matplotlib import pyplot as plt
from scipy.stats import friedmanchisquare

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.plotting.consts import (
    CLUSTERER_LABELS,
    DATASET_ORDER,
    categorical_palette,
    clusterer_label_order,
    dataset_display_name,
    figure_size,
    save_figure,
    setup_publication_style,
)
from vmf_hac.plotting.embedding_models import MODEL_LABELS

# Studentized range statistic divided by sqrt(2) at alpha = 0.05, indexed by the number
# of compared algorithms (Demsar 2006, Table 5).
NEMENYI_Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


def load_scores() -> pd.DataFrame:
    """One score per (dataset, embedding model, algorithm).

    The k grid differs between datasets, so the ARI curve is summarised by its area
    under the curve normalised by the width of the k range; this is the k-averaged ARI
    and is comparable across datasets.
    """
    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_embedding_models.csv")
    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    df["clusterer_label"] = df["clusterer_name"].map(CLUSTERER_LABELS).fillna(df["clusterer_name"])
    df["model_label"] = df["model_name"].map(MODEL_LABELS).fillna(df["model_name"])

    rows: list[dict[str, object]] = []
    for (dataset, model, clusterer), frame in df.groupby(["dataset_label", "model_label", "clusterer_label"]):
        curve = frame.groupby("n_clusters")["v_measure"].mean().sort_index()
        k = curve.index.to_numpy(dtype=float)
        span = k.max() - k.min()
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "clusterer": clusterer,
                "auc": float(np.trapezoid(curve.to_numpy(), k) / span),
                "v_measure_at_true": float(curve.iloc[int(np.abs(k - frame["n_true_clusters"].iloc[0]).argmin())]),
            }
        )

    scores = pd.DataFrame(rows)
    scores["rank"] = scores.groupby(["dataset", "model"])["auc"].rank(ascending=False)
    return scores


def _orders(scores: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    clusterer_order = clusterer_label_order(set(scores["clusterer"]))
    model_order = [label for label in MODEL_LABELS.values() if label in set(scores["model"])]
    dataset_order = [label for label in DATASET_ORDER if label in set(scores["dataset"])]
    return clusterer_order, model_order, dataset_order


def plot_rank_slopegraph(scores: pd.DataFrame) -> None:
    """Mean rank per algorithm under each embedding model.

    Flat lines mean the embedding model only shifts the level of ARI; crossings mean it
    actually reorders the algorithms.
    """
    clusterer_order, model_order, _ = _orders(scores)
    palette = categorical_palette(clusterer_order)
    table = scores.pivot_table(index="clusterer", columns="model", values="rank").reindex(
        index=clusterer_order, columns=model_order
    )

    fig, ax = plt.subplots(figsize=figure_size(columns=1, aspect=0.85), layout="constrained")
    x = np.arange(len(model_order), dtype=float)
    for clusterer in clusterer_order:
        ax.plot(x, table.loc[clusterer], color=palette[clusterer], marker="o", markersize=2.5, linewidth=0.9)

    # Labelling the end points instead of using a legend keeps the lines traceable.
    for clusterer in clusterer_order:
        ax.annotate(
            clusterer,
            (x[-1], table.loc[clusterer, model_order[-1]]),
            xytext=(3, 0),
            textcoords="offset points",
            color=palette[clusterer],
            va="center",
            fontsize=6,
        )

    ax.set_xticks(x, model_order)
    ax.set_xlim(-0.15, x[-1] + 0.95)
    ax.invert_yaxis()
    ax.set_ylabel("mean rank over datasets (1 = best)")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, ROOT_DIR / "results" / "plots" / "embedding_models_rank_slope.pdf")


def _cliques(names: list[str], values: np.ndarray, cd: float) -> list[tuple[int, int]]:
    """Maximal groups of consecutive algorithms whose mean ranks differ by less than CD."""
    groups: list[tuple[int, int]] = []
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and values[j + 1] - values[i] < cd:
            j += 1
        if j > i:
            groups.append((i, j))
    return [g for g in groups if not any(g != o and o[0] <= g[0] and g[1] <= o[1] for o in groups)]


def plot_critical_difference(scores: pd.DataFrame) -> float:
    """Demsar-style critical difference diagram over the dataset x model problems."""
    clusterer_order, _, _ = _orders(scores)
    matrix = scores.pivot_table(index=["dataset", "model"], columns="clusterer", values="auc")[clusterer_order]
    n_problems, n_algorithms = matrix.shape
    friedman = friedmanchisquare(*(matrix[c].to_numpy() for c in clusterer_order))
    cd = NEMENYI_Q05[n_algorithms] * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_problems))

    mean_ranks = matrix.rank(axis=1, ascending=False).mean(axis=0).sort_values()
    names = list(mean_ranks.index)
    values = mean_ranks.to_numpy()

    fig, ax = plt.subplots(figsize=figure_size(columns=1, aspect=0.62), layout="constrained")
    lo, hi = np.floor(values.min()), np.ceil(values.max())
    ax.set_xlim(lo - 0.2, hi + 0.2)
    ax.set_ylim(-(len(names) // 2 + 2.2), 2.4)
    ax.axis("off")

    ax.plot([lo, hi], [0, 0], color="black", linewidth=0.8)
    for tick in np.arange(lo, hi + 0.5, 0.5):
        ax.plot([tick, tick], [0, 0.12], color="black", linewidth=0.8)
        if float(tick).is_integer():
            ax.text(tick, 0.22, f"{tick:g}", ha="center", va="bottom", fontsize=6)

    half = int(np.ceil(len(names) / 2))
    for i, (name, value) in enumerate(zip(names, values, strict=True)):
        left = i < half
        depth = -(i + 1) if left else -(len(names) - i)
        edge = lo - 0.2 if left else hi + 0.2
        ax.plot([value, value, edge], [0, depth, depth], color="black", linewidth=0.6)
        ax.text(
            edge,
            depth,
            f"{name} ",
            ha="right" if left else "left",
            va="center",
            fontsize=6,
        )

    for level, (i, j) in enumerate(_cliques(names, values, cd)):
        y = -0.28 - 0.22 * level
        ax.plot([values[i] - 0.03, values[j] + 0.03], [y, y], color="black", linewidth=2.2, solid_capstyle="butt")

    ax.plot([lo, lo + cd], [1.15, 1.15], color="black", linewidth=0.8)
    for end in (lo, lo + cd):
        ax.plot([end, end], [1.08, 1.22], color="black", linewidth=0.8)
    ax.text(lo + cd / 2, 1.3, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=6)
    ax.text(
        (lo + hi) / 2,
        2.0,
        f"$N = {n_problems}$ dataset $\\times$ model pairs, Friedman $p = {friedman.pvalue:.1e}$",
        ha="center",
        va="bottom",
        fontsize=6,
    )

    save_figure(fig, ROOT_DIR / "results" / "plots" / "embedding_models_cd.pdf")
    return cd


def write_rank_table(scores: pd.DataFrame) -> None:
    clusterer_order, model_order, _ = _orders(scores)
    auc = scores.pivot_table(index="clusterer", columns="model", values="auc").reindex(
        index=clusterer_order, columns=model_order
    )
    rank = scores.pivot_table(index="clusterer", columns="model", values="rank").reindex(
        index=clusterer_order, columns=model_order
    )

    table = pd.concat({"$\\overline{\\text{ARI}}_k$": auc, "rank": rank}, axis=1)
    table[("", "mean rank")] = rank.mean(axis=1)
    table = table.sort_values(("", "mean rank"))

    formatted = table.map(lambda v: f"{v:.3f}")
    for column in auc.columns:
        best = table[("$\\overline{\\text{ARI}}_k$", column)].idxmax()
        value = formatted.loc[best, ("$\\overline{\\text{ARI}}_k$", column)]
        formatted.loc[best, ("$\\overline{\\text{ARI}}_k$", column)] = f"\\textbf{{{value}}}"
    for column in [*[("rank", c) for c in rank.columns], ("", "mean rank")]:
        best = table[column].idxmin()
        formatted.loc[best, column] = f"\\textbf{{{formatted.loc[best, column]}}}"

    formatted.index.name = "Algorithm"
    latex = formatted.to_latex(escape=False, multicolumn_format="c")
    os.makedirs(ROOT_DIR / "results" / "tables", exist_ok=True)
    with open(ROOT_DIR / "results" / "tables" / "embedding_models_rank.tex", "w", encoding="utf-8") as file:
        file.write(latex)


def main():
    setup_publication_style()
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)
    scores = load_scores()
    plot_rank_slopegraph(scores)
    cd = plot_critical_difference(scores)
    write_rank_table(scores)
    print(f"critical difference (alpha=0.05): {cd:.3f}")


if __name__ == "__main__":
    main()
