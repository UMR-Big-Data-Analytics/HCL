from collections.abc import Sequence

import seaborn as sns

PALETTE_NAME = "colorblind"

DATASET_NAME_MAP = {
    "mteb/WikiCitiesClustering": "WikiCities",
    "mteb/llm-eval-banking77": "Banking77",
    "mteb/llm-eval-dbpedia_14": "DBPedia",
    "sklearn/20newsgroups": "20Newsgroups",
}

DATASET_ORDER = ["DBPedia", "Banking77", "20Newsgroups", "WikiCities"]
_DATASET_COLORS = sns.color_palette(PALETTE_NAME, n_colors=len(DATASET_ORDER))
DATASET_PALETTE = dict(zip(DATASET_ORDER, _DATASET_COLORS, strict=False))


def categorical_palette(labels: Sequence[object]) -> dict[object, tuple[float, float, float]]:
    colors = sns.color_palette(PALETTE_NAME, n_colors=len(labels))
    return dict(zip(labels, colors, strict=False))
