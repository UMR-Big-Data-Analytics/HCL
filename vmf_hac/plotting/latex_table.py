from typing import Literal

import pandas as pd


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def style_top3_latex(
    table: pd.DataFrame,
    *,
    std_table: pd.DataFrame | None = None,
    mean_index: str | None = "Mean",
    rank_within: Literal["row", "column"] = "row",
    index_name: str = "Dataset",
    std_mode: Literal["inline", "none"] = "inline",
) -> str:
    """Render *table* as a LaTeX booktabs table with the top-3 values highlighted.

    - 1st: bold + solid underline (``\\uline``)
    - 2nd: bold + dashed underline (``\\dashuline``)
    - 3rd: bold + dotted underline (``\\dotuline``)

    If *std_table* is provided, *std_mode* controls whether each cell is rendered as
    ``mean ± std`` (``inline``) or just the mean (``none``). Ranking is always based on
    the mean values in *table*.

    *rank_within* selects whether the top-3 are picked across each row (the default,
    for tables whose columns are the compared algorithms) or down each column (for
    the transposed layout). A *mean_index* row never contributes to column ranks.

    If *mean_index* is given and a row with that label exists, it is rendered as
    ``\\textbf{Mean}`` and preceded by a ``\\midrule``.
    """
    if std_mode not in {"inline", "none"}:
        msg = f"Unsupported std_mode: {std_mode}"
        raise ValueError(msg)

    algorithm_columns = list(table.columns)

    formatted = table.copy().astype(object)
    for col in algorithm_columns:
        for idx in table.index:
            mean_val = table.loc[idx, col]
            if std_table is not None and std_mode == "inline":
                std_val = std_table.loc[idx, col]
                formatted.loc[idx, col] = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"
            else:
                formatted.loc[idx, col] = f"{mean_val:.3f}"

    if rank_within == "row":
        ranks = table[algorithm_columns].rank(axis=1, ascending=False, method="min")
    else:
        basis = table.drop(index=mean_index) if mean_index in table.index else table
        ranks = basis[algorithm_columns].rank(axis=0, ascending=False, method="min")

    underlines = {1: "uline", 2: "dashuline", 3: "dotuline"}
    for idx in ranks.index:
        for col in algorithm_columns:
            command = underlines.get(int(ranks.loc[idx, col]))
            if command is not None:
                formatted.loc[idx, col] = f"\\textbf{{\\{command}{{{formatted.loc[idx, col]}}}}}"

    formatted.columns = [latex_escape(str(col)) for col in formatted.columns]
    formatted.index = [latex_escape(str(idx)) for idx in formatted.index]

    if mean_index is not None:
        escaped_mean = latex_escape(mean_index)
        if escaped_mean in formatted.index:
            formatted.rename(index={escaped_mean: r"\textbf{Mean}"}, inplace=True)

    formatted.index.name = index_name
    latex = formatted.to_latex(escape=False, float_format=None)

    if mean_index is not None:
        latex = latex.replace("\n\\textbf{Mean} &", "\n\\midrule\n\\textbf{Mean} &", 1)

    return latex
