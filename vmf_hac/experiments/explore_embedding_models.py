import math
import os
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from vmf_hac.baselines import VonMisesFisherMixture
from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity import DatasetManager, TextDatasets
from vmf_hac.entity.clusterer import Clusterer
from vmf_hac.experiments.methods import METHODS
from vmf_hac.utils import evaluate
from vmf_hac.utils.config import get_config
from vmf_hac.utils.parallel import gather, task


@task
def run(clusterer: Clusterer, x: np.ndarray, y: np.ndarray, dataset_name: str, model_name: str) -> dict[str, Any]:
    if isinstance(clusterer, AgglomerativeClustering):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.linkage}"
    elif isinstance(clusterer, VonMisesFisherMixture):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.posterior_type}"
    else:
        clusterer_name = clusterer.__class__.__name__
    result = evaluate(clusterer, dataset_name, x, y, clusterer_name)
    return {**result.to_dict(), "model_name": model_name}


def main():
    config = get_config()
    jobs = []

    datasets = [
        (TextDatasets.DBPEDIA_14, "DBPedia"),
        (TextDatasets.BUILT_BENCH_CLUSTERING_P2P, "BuiltBenchP2P"),
        (TextDatasets.CLUSTREC_COVID, "ClusTREC-Covid"),
        (TextDatasets.WIKICITIES, "WikiCities"),
    ]
    for (dataset, dataset_name), model_name in product(datasets, config["models"]):
        dataset = DatasetManager(model_name).get(dataset)
        x, y = dataset.embeddings, dataset.labels
        true_k = np.unique(y).shape[0]
        for _, method_factory in METHODS:
            for k in np.arange(2, 2 * true_k, math.ceil(true_k / 10)):
                method = method_factory(k)
                jobs.append(run(method, x, y, dataset_name, model_name))

    results = gather(jobs, show_progress=True)
    df = pd.DataFrame(results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_embedding_models.csv", index=False)


if __name__ == "__main__":
    main()
