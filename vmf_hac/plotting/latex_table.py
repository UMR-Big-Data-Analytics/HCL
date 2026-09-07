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
) -> str:
    """Render *table* as a LaTeX booktabs table with the top-3 values per row highlighted.

    - 1st: bold + solid underline (``\\uline``)
    - 2nd: bold + dashed underline (``\\dashuline``)
    - 3rd: bold + dotted underline (``\\dotuline``)

    If *std_table* is provided (same shape as *table*), each cell is rendered as
    ``mean ± std``.  Ranking is always based on the mean values in *table*.

    If *mean_index* is given and a row with that label exists, it is rendered as
    ``\\textbf{Mean}`` and preceded by a ``\\midrule``.
    """
    algorithm_columns = list(table.columns)

    formatted = table.copy().astype(object)
    for col in algorithm_columns:
        for idx in table.index:
            mean_val = table.loc[idx, col]
            if std_table is not None:
                std_val = std_table.loc[idx, col]
                formatted.loc[idx, col] = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"
            else:
                formatted.loc[idx, col] = f"{mean_val:.3f}"

    for idx in formatted.index:
        row_values = table.loc[idx, algorithm_columns]
        row_ranks = row_values.rank(ascending=False, method="min")
        for col in algorithm_columns:
            rank = row_ranks.loc[col]
            value = formatted.loc[idx, col]
            if rank == 1:
                formatted.loc[idx, col] = f"\\textbf{{\\uline{{{value}}}}}"
            elif rank == 2:
                formatted.loc[idx, col] = f"\\textbf{{\\dashuline{{{value}}}}}"
            elif rank == 3:
                formatted.loc[idx, col] = f"\\textbf{{\\dotuline{{{value}}}}}"

    formatted.columns = [latex_escape(str(col)) for col in formatted.columns]
    formatted.index = [latex_escape(str(idx)) for idx in formatted.index]

    if mean_index is not None:
        escaped_mean = latex_escape(mean_index)
        if escaped_mean in formatted.index:
            formatted.rename(index={escaped_mean: r"\textbf{Mean}"}, inplace=True)

    formatted.index.name = "Dataset"
    latex = formatted.to_latex(escape=False, float_format=None)

    if mean_index is not None:
        latex = latex.replace("\n\\textbf{Mean} &", "\n\\midrule\n\\textbf{Mean} &", 1)

    return latex
