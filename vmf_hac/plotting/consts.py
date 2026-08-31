import shutil
from collections.abc import Sequence
from pathlib import Path
from warnings import warn

import figure_scale as fs
import scienceplots  # noqa: F401  (registers the "science" matplotlib style)
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from vmf_hac.entity.dataset import DatasetFactory

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
    "VmfHAC": "vMF-HAC",
    "AgglomerativeClustering_ward": "Ward",
    "AgglomerativeClustering_single": "Single",
    "AgglomerativeClustering_complete": "Complete",
    "AgglomerativeClustering_average": "Average",
    "VonMisesFisherMixture_soft": "moVMF",
    "SphericalKMeans": "Spherical KM",
    "KMeans": "K-Means",
    "SpectralClustering": "Spectral",
}


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


def categorical_palette(labels: Sequence[object]) -> dict[object, tuple[float, float, float]]:
    colors = sns.color_palette(PALETTE_NAME, n_colors=len(labels))
    return dict(zip(labels, colors, strict=False))


def figure_width_pt(columns: int = 1) -> float:
    return COLUMN_WIDTH_PT * columns + COLUMN_SEP_PT * (columns - 1)


def figure_size(*, columns: int = 1, aspect: float = DEFAULT_FIGURE_ASPECT) -> fs.FigureScale:
    return fs.FigureScale(units="pt", width=figure_width_pt(columns), aspect=aspect)


def page_aspect_limit(columns: int = 1) -> float:
    return TEXT_HEIGHT_PT / figure_width_pt(columns)


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
