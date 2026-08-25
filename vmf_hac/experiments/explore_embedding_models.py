import gc
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
from vmf_hac.entity.dataset import DatasetFactory
from vmf_hac.experiments.methods import METHODS
from vmf_hac.utils import evaluate
from vmf_hac.utils.config import get_config
from vmf_hac.utils.data import random_subset
from vmf_hac.utils.parallel import gather, task

METHOD_FACTORIES = dict(METHODS)


@task
def run(
    method_name: str,
    dataset_id: str,
    dataset_name: str,
    model_name: str,
    *,
    split: int | None = None,
    n_clusters: int | None = None,
) -> dict[str, Any]:
    method_factory = METHOD_FACTORIES[method_name]
    dataset = DatasetManager(model_name).get(DatasetFactory.from_string(dataset_id), skip_cache=True)
    x, y = dataset.embeddings, dataset.labels
    if split is not None:
        x, y = random_subset(x, y, n=1000, seed=split)
        gc.collect()
    if n_clusters is None:
        n_clusters = int(np.unique(y).shape[0])
    clusterer: Clusterer = method_factory(n_clusters)
    if isinstance(clusterer, AgglomerativeClustering):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.linkage}"
    elif isinstance(clusterer, VonMisesFisherMixture):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.posterior_type}"
    else:
        clusterer_name = clusterer.__class__.__name__
    result = evaluate(clusterer, dataset_name, x, y, clusterer_name)
    return {**result.to_dict(), "model_name": model_name, "split": split}


def embedding_models_over_k():
    config = get_config()
    jobs = []

    datasets = [
        (TextDatasets.DBPEDIA_14, "DBPedia"),
        (TextDatasets.BUILT_BENCH_CLUSTERING_P2P, "BuiltBenchP2P"),
        (TextDatasets.CLUSTREC_COVID, "ClusTREC-Covid"),
        (TextDatasets.WIKICITIES, "WikiCities"),
    ]
    for (dataset, dataset_name), model_name in product(datasets, config["models"]):
        # Load only labels in the main process (embeddings stay mmap'd on demand in workers).
        y = DatasetManager(model_name).get(dataset).labels
        true_k = np.unique(y).shape[0]
        # Free the label array; the TextDataset shell stays cached but is cheap.
        del y
        for method_name, _ in METHODS:
            for k in np.arange(2, 2 * true_k, math.ceil(true_k / 10)):
                jobs.append(
                    run(
                        method_name=method_name,
                        dataset_id=dataset.value.technical_name,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        n_clusters=int(k),
                    )
                )

    # Use the default loky (process) backend so workers are not limited by the GIL.
    # With mmap_mode='r' in TextDataset.embeddings the OS page cache is shared
    # across worker processes, so the embedding arrays are not duplicated in RAM.
    results = gather(jobs, show_progress=True)
    df = pd.DataFrame(results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_embedding_models_k.csv", index=False)


def embedding_models():
    config = get_config()
    jobs = []
    splits = 5
    for dataset_name, model_name in product(config["datasets"], config["models"]):
        for split in range(splits):
            for method_name, _ in METHODS:
                jobs.append(
                    run(
                        method_name=method_name,
                        dataset_id=dataset_name,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        split=split,
                    )
                )

    results = gather(jobs, show_progress=True)
    df = pd.DataFrame(results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_embedding_models.csv", index=False)


if __name__ == "__main__":
    embedding_models()
    # embedding_models_over_k()
