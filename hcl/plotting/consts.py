import shutil
from collections.abc import Sequence
from pathlib import Path
from warnings import warn

import figure_scale as fs
import scienceplots  # noqa: F401  (registers the "science" matplotlib style)
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from hcl.entity.dataset import DatasetFactory

PALETTE_NAME = "colorblind"
PT_PER_INCH = 72.27
LATEX_PREAMBLE = "\n".join(
    [
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
    ]
)

COLUMN_WIDTH_PT = 232.75394
COLUMN_SEP_PT = 22.31
TEXT_HEIGHT_PT = 666.0

PUBLICATION_FONT_SIZE_PT = 8
DEFAULT_FIGURE_ASPECT = 3 / 4
SQUARE_FIGURE_ASPECT = 1.0

_DATASET_NAME_MAP: dict[str, str] | None = None

DATASET_ORDER = ["DBPedia", "BuiltBenchP2P", "ClusTREC-Covid", "Wikicities"]

# Canonical algorithm name → display label mapping.
# Insertion order defines the preferred display order across all plots and tables.
CLUSTERER_LABELS: dict[str, str] = {
    "HclHAC": "HCL",
    "AgglomerativeClustering_ward": "Ward",
    "AgglomerativeClustering_single": "Single",
    "AgglomerativeClustering_average": "Average",
    "AgglomerativeClustering_complete": "Complete",
    "KMeans": "K-Means",
    "SphericalKMeans": "Spherical KM",
    "SpectralClustering": "Spectral",
    "VonMisesFisherMixture_soft": "moVMF",
    "umap_hdbscan": "HDBSCAN$^+$",
}

# One distinct colour per clustering method, shared by the matplotlib figures and the
# critdd diagrams so a method keeps its identity across every figure in the paper.
CLUSTERER_COLORS: dict[str, str] = {
    "HCL": "009E73",
    "Ward": "D55E00",
    "Single": "999999",
    "Average": "E69F00",
    "Complete": "56B4E9",
    "K-Means": "0072B2",
    "Spherical KM": "CC79A7",
    "Spectral": "8C564B",
    "moVMF": "7E2F8E",
    "HDBSCAN$^+$": "FC0352",
}

# Marks are only used by the critdd diagrams, which draw discrete points rather than lines.
CLUSTERER_TIKZ_MARKS: dict[str, str] = {
    "HCL": "*",
    "Ward": "square*",
    "Single": "otimes*",
    "Average": "triangle*",
    "Complete": "diamond*",
    "K-Means": "pentagon*",
    "Spherical KM": "square*",
    "Spectral": "*",
    "moVMF": "asterisk",
    "HDBSCAN$^+$": "oplus*",
}

HDBSCAN_LABEL = "HDBSCAN$^+$"


def critdd_color_name(label: object) -> str:
    """Return a LaTeX-safe colour name for *label* (TeX macros cannot contain punctuation)."""
    return "cd" + "".join(character for character in str(label) if character.isalnum())


def critdd_preamble(labels: Sequence[object]) -> str:
    r"""Return ``\definecolor`` declarations for *labels* in the shared palette."""
    return "\n".join(
        f"\\definecolor{{{critdd_color_name(label)}}}{{HTML}}{{{CLUSTERER_COLORS[str(label)]}}}"
        for label in labels
        if str(label) in CLUSTERER_COLORS
    )


def critdd_cycle_list(labels: Sequence[object]) -> str:
    """Return a pgfplots cycle list that matches the shared colours and marks."""
    entries = []
    for label in labels:
        color = critdd_color_name(label)
        mark = CLUSTERER_TIKZ_MARKS.get(str(label), "*")
        entries.append(f"{{{color},mark={mark}}}")
    return ",".join(entries)


def clusterer_label_order(present: set[str]) -> list[str]:
    """Return display labels in canonical order, filtered to those present in *present*.

    *present* may contain either algorithm names (keys of CLUSTERER_LABELS) or
    display labels (values).  Unknown entries are appended in sorted order.
    """
    known_labels = list(CLUSTERER_LABELS.values())
    # Support callers that pass a set of display labels directly.
    if present and next(iter(present)) in known_labels:
        order = [label for label in known_labels if label in present]
        order.extend(sorted(present - set(order)))
    else:
        order = [CLUSTERER_LABELS[name] for name in CLUSTERER_LABELS if name in present]
        extra_labels = present - set(CLUSTERER_LABELS.keys())
        order.extend(sorted(extra_labels - set(order)))
    return order


_DATASET_COLORS = sns.color_palette(PALETTE_NAME, n_colors=len(DATASET_ORDER))
DATASET_PALETTE = dict(zip(DATASET_ORDER, _DATASET_COLORS, strict=False))


def dataset_display_name(dataset_name: object) -> str:
    """Return the canonical display name for a dataset identifier or label."""
    name = str(dataset_name)
    return DatasetFactory.get_dataset_name_map().get(name, name.rsplit("/", maxsplit=1)[-1])


def categorical_palette(labels: Sequence[object]) -> dict[object, tuple[float, float, float] | str]:
    palette: dict[object, tuple[float, float, float] | str] = {}
    fallback_labels: list[object] = []
    for label in labels:
        if str(label) in CLUSTERER_COLORS:
            palette[label] = "#" + CLUSTERER_COLORS[str(label)]
        else:
            fallback_labels.append(label)

    if fallback_labels:
        colors = sns.color_palette(PALETTE_NAME, n_colors=len(fallback_labels))
        palette.update(dict(zip(fallback_labels, colors, strict=False)))

    return palette


def figure_width_pt(columns: int = 1, scale: float = 1.0) -> float:
    r"""Figure width in pt.

    *scale* is the fraction of the width used in LaTeX, e.g. ``0.7`` for
    ``\includegraphics[width=0.7\linewidth]``. The figure is created at that physical
    width so LaTeX includes it 1:1 and fonts and lines keep their nominal sizes.
    """
    if not 0 < scale <= 1:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    return (COLUMN_WIDTH_PT * columns + COLUMN_SEP_PT * (columns - 1)) * scale


def figure_size(*, columns: int = 1, aspect: float = DEFAULT_FIGURE_ASPECT, scale: float = 1.0) -> fs.FigureScale:
    return fs.FigureScale(units="pt", width=figure_width_pt(columns, scale), aspect=aspect)


def page_aspect_limit(columns: int = 1, scale: float = 1.0) -> float:
    return TEXT_HEIGHT_PT / figure_width_pt(columns, scale)


def latex_available() -> bool:
    return all(shutil.which(binary) is not None for binary in ("latex", "dvipng"))


def setup_publication_style(
    *,
    columns: int = 1,
    aspect: float = DEFAULT_FIGURE_ASPECT,
    use_latex: bool = True,
) -> None:
    plt.style.use("science")
    rc_params: dict[str, object] = {
        "figure.figsize": figure_size(columns=columns, aspect=aspect),
        "axes.labelsize": PUBLICATION_FONT_SIZE_PT,
        "axes.titlesize": PUBLICATION_FONT_SIZE_PT,
        "font.size": PUBLICATION_FONT_SIZE_PT,
        "legend.fontsize": PUBLICATION_FONT_SIZE_PT,
        "xtick.labelsize": PUBLICATION_FONT_SIZE_PT,
        "ytick.labelsize": PUBLICATION_FONT_SIZE_PT,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        # The "science" style crops to the drawn content, which makes the saved figure
        # narrower than the column; keep the exact figure size instead.
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "font.family": "serif",
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": LATEX_PREAMBLE,
    }

    if use_latex and not latex_available():
        warn("LaTeX not available. Using default matplotlib text rendering.", stacklevel=2)
        use_latex = False
    rc_params["text.usetex"] = use_latex
    if use_latex:
        rc_params["text.latex.preamble"] = LATEX_PREAMBLE

    plt.rcParams.update(rc_params)  # ty:ignore[no-matching-overload]


def save_figure(fig: Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backend = "pgf" if plt.rcParams["text.usetex"] else None
    fig.savefig(path, backend=backend)
