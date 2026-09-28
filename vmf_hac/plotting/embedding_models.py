import os
import subprocess

import numpy as np
import pandas as pd
import scienceplots  # noqa
from critdd import Diagrams
from matplotlib import pyplot as plt

from vmf_hac.definitions import ROOT_DIR
from vmf_hac.plotting.consts import (
    CLUSTERER_LABELS,
    DATASET_ORDER,
    DEFAULT_FIGURE_ASPECT,
    categorical_palette,
    clusterer_label_order,
    critdd_cycle_list,
    critdd_preamble,
    dataset_display_name,
    figure_size,
    page_aspect_limit,
    save_figure,
    setup_publication_style,
)
from vmf_hac.plotting.latex_table import style_top3_latex

MODEL_LABELS = {
    "intfloat/multilingual-e5-large": "multi-e5-large",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-Emb-8B",
    "codefuse-ai/F2LLM-v2-14B": "F2LLM-v2-14B",
}

HIGHLIGHT_CLUSTERER = CLUSTERER_LABELS["VmfHAC"]


def _prepare_embedding_model_df() -> pd.DataFrame:
    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_embedding_models.csv")
    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    df["clusterer_label"] = df["clusterer_name"].map(CLUSTERER_LABELS).fillna(df["clusterer_name"])
    df["model_label"] = df["model_name"].map(MODEL_LABELS).fillna(df["model_name"])
    df["v_measure_effective"] = df["v_measure"] * (1 - df["noise_fraction"])
    return df


def _model_order(df: pd.DataFrame) -> list[str]:
    return [MODEL_LABELS[name] for name in MODEL_LABELS if name in set(df["model_name"].unique())]


def _clusterer_order(df: pd.DataFrame) -> list[str]:
    return clusterer_label_order(set(df["clusterer_label"].unique()))


def _save_critdd_diagram(diagram: Diagrams, path: str, treatment_names: list[str]) -> None:
    tikz = diagram.to_str(
        alpha=0.05,
        adjustment="holm",
        reverse_x=True,
        as_document=True,
        preamble=critdd_preamble(treatment_names),
        axis_options={"cycle list": critdd_cycle_list(treatment_names)},
    )
    output = ROOT_DIR / "results" / "plots" / path
    tex_path = output.with_suffix(".critdd.tex")
    tex_path.write_text(tikz)
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            f"-output-directory={tex_path.parent}",
            str(tex_path),
        ],
        check=True,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    pdf_path.replace(output)


def _plot_grid(df: pd.DataFrame, *, normalize: bool, filename: str) -> None:
    clusterer_order = clusterer_label_order(set(df["clusterer_name"].unique()))
    clusterer_palette = categorical_palette(clusterer_order)
    model_order = [MODEL_LABELS[name] for name in MODEL_LABELS if name in set(df["model_name"].unique())]
    dataset_order = [label for label in DATASET_ORDER if label in set(df["dataset_label"])]

    # One column per embedding model, one row per dataset: the k grid differs between
    # datasets, so their curves cannot share a panel.
    cols = len(model_order)
    rows = len(dataset_order)
    aspect = min(0.8 * rows / cols, page_aspect_limit(columns=2))
    fig = plt.figure(figsize=figure_size(columns=2, aspect=aspect, scale=0.8), layout="constrained")
    axes = np.atleast_2d(fig.subplots(rows, cols, sharex="row", sharey="row")).reshape(rows, cols)

    legend_handles: list = []
    legend_labels: list = []

    for row, dataset_label in enumerate(dataset_order):
        dataset_df = df[df["dataset_label"] == dataset_label]
        n_true_clusters = int(dataset_df.iloc[0]["n_true_clusters"])
        for col, model_label in enumerate(model_order):
            ax = axes[row, col]
            panel_df = dataset_df[dataset_df["model_label"] == model_label].sort_values("n_clusters").copy()
            if normalize:
                # Dividing by the best ARI reached in the panel removes the level shift the
                # embedding model induces, so only the relative ordering remains visible.
                panel_df["v_measure_effective"] = (
                    panel_df["v_measure_effective"] / panel_df["v_measure_effective"].max()
                )
            for clusterer_label in clusterer_order:
                clusterer_frame = panel_df[panel_df["clusterer_label"] == clusterer_label]
                if clusterer_frame.empty:
                    continue
                ax.plot(
                    clusterer_frame["n_clusters"],
                    clusterer_frame["v_measure_effective"],
                    linewidth=0.9,
                    color=clusterer_palette[clusterer_label],
                    label=clusterer_label,
                )
            ax.axvline(float(n_true_clusters), color="red", linestyle=":", linewidth=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if row == 0:
                ax.set_title(model_label)
            if col == 0:
                ax.set_ylabel(dataset_label)

            handles, labels = ax.get_legend_handles_labels()
            if not legend_handles and labels:
                legend_handles, legend_labels = handles, labels
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    for ax in axes[-1, :]:
        ax.set_xlabel("$k$")

    fig.supylabel("V-Measure / panel max" if normalize else "V-Measure")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="outside lower center",
            ncol=min(len(legend_labels), 5),
            frameon=False,
        )

    save_figure(fig, ROOT_DIR / "results" / "plots" / filename)


def embedding_models_over_k():
    df = pd.read_csv(ROOT_DIR / "results" / "data" / "explore_embedding_models_k.csv")
    df["dataset_label"] = df["dataset_name"].map(dataset_display_name)
    df["clusterer_label"] = df["clusterer_name"].map(CLUSTERER_LABELS).fillna(df["clusterer_name"])
    df["model_label"] = df["model_name"].map(MODEL_LABELS).fillna(df["model_name"])

    _plot_grid(df, normalize=False, filename="embedding_models.pdf")
    _plot_grid(df, normalize=True, filename="embedding_models_normalized.pdf")


def _model_tick_labels(model_order: list[str]) -> list[str]:
    return [f"{label}" for label in model_order]


def _rank_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and split, rank the methods, then average the ranks per dataset."""
    ranks = df[["dataset_label", "split", "model_label", "clusterer_label", "v_measure_effective"]].copy()
    ranks["rank"] = ranks.groupby(["dataset_label", "split", "model_label"])["v_measure_effective"].rank(
        ascending=False, method="average"
    )
    return ranks.groupby(["dataset_label", "model_label", "clusterer_label"])["rank"].mean().reset_index()


def _plot_level_and_order(df: pd.DataFrame) -> None:
    """Two panels: the embedding model shifts the score level but keeps the method order."""
    model_order = _model_order(df)
    clusterer_order = _clusterer_order(df)
    palette = categorical_palette(clusterer_order)

    score = df.groupby(["dataset_label", "model_label", "clusterer_label"])["v_measure_effective"].mean().reset_index()
    level = score.pivot_table(index="clusterer_label", columns="model_label", values="v_measure_effective").reindex(
        index=clusterer_order, columns=model_order
    )
    order = (
        _rank_table(df)
        .pivot_table(index="clusterer_label", columns="model_label", values="rank")
        .reindex(index=clusterer_order, columns=model_order)
    )

    fig = plt.figure(
        figsize=figure_size(columns=2, aspect=DEFAULT_FIGURE_ASPECT * 0.5, scale=0.8),
        layout="constrained",
    )
    left, right = fig.subplots(1, 2)
    x = np.arange(len(model_order))

    for ax, table in ((left, level), (right, order)):
        for clusterer_label in clusterer_order:
            ax.plot(
                x,
                table.loc[clusterer_label].to_numpy(),
                linewidth=1.0,
                color=palette[clusterer_label],
                label=clusterer_label,
                marker="o",
                zorder=2,
            )
        ax.set_xticks(x, model_order)
        ax.set_xlim(-0.25, len(model_order) - 0.75)
        # ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(True, axis="y", alpha=0.3)

    left.set_ylabel("Mean V-Measure")
    # left.set_title("(a) Absolute Quality", fontsize=PUBLICATION_FONT_SIZE_PT)
    right.set_ylabel("Mean Rank")
    # right.set_title("(b) Relative Ranking", fontsize=PUBLICATION_FONT_SIZE_PT)
    right.set_yticks(np.arange(1, len(clusterer_order) + 1))
    right.invert_yaxis()

    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save_figure(fig, ROOT_DIR / "results" / "plots" / "embedding_models_v_measure.pdf")
    plt.close(fig)


def _write_level_table(df: pd.DataFrame) -> None:
    """Rows are methods and columns embedding models, so the top-3 is read down each column."""
    model_order = _model_order(df)
    clusterer_order = _clusterer_order(df)
    dataset_means = (
        df.groupby(["dataset_label", "model_label", "clusterer_label"])["v_measure_effective"].mean().reset_index()
    )
    pivot_kwargs = {"index": "clusterer_label", "columns": "model_label", "values": "v_measure_effective"}
    mean_table = dataset_means.pivot_table(**pivot_kwargs, aggfunc="mean").reindex(  # ty:ignore
        index=clusterer_order, columns=model_order
    )
    std_table = dataset_means.pivot_table(**pivot_kwargs, aggfunc="std").reindex(  # ty:ignore
        index=clusterer_order, columns=model_order
    )

    latex = style_top3_latex(
        mean_table,
        std_table=std_table,
        mean_index=None,
        rank_within="column",
        index_name="Method",
        std_mode="none",
    )

    path = ROOT_DIR / "results" / "tables" / "embedding_models_v_measure_effective_mean.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")

    appendix_latex = style_top3_latex(
        mean_table,
        std_table=std_table,
        mean_index=None,
        rank_within="column",
        index_name="Method",
        std_mode="inline",
    )

    appendix_path = ROOT_DIR / "results" / "tables" / "embedding_models_v_measure_mean_std.tex"
    appendix_path.write_text(appendix_latex, encoding="utf-8")


def _write_cd_diagram(df: pd.DataFrame) -> None:
    """One critical-difference diagram per embedding model, over all dataset/split observations."""
    model_order = _model_order(df)
    clusterer_order = _clusterer_order(df)
    scores = []
    for model_label in model_order:
        panel = df[df["model_label"] == model_label][
            ["dataset_label", "split", "clusterer_label", "v_measure_effective"]
        ]
        pivot = (
            panel.pivot(index=["dataset_label", "split"], columns="clusterer_label", values="v_measure_effective")
            .reindex(columns=clusterer_order)
            .dropna()
        )
        scores.append(pivot.to_numpy())
    diagram = Diagrams(
        scores,
        diagram_names=model_order,
        treatment_names=clusterer_order,
        maximize_outcome=True,
    )
    _save_critdd_diagram(diagram, "embedding_models_cd_2d.pdf", clusterer_order)


def embedding_models():
    df = _prepare_embedding_model_df()
    _plot_level_and_order(df)
    _write_level_table(df)
    _write_cd_diagram(df)


if __name__ == "__main__":
    setup_publication_style()
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)
    embedding_models()
    # embedding_models_over_k()
