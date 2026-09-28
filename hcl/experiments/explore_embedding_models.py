import math
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import Pipeline

from hcl.baselines import VonMisesFisherMixture
from hcl.definitions import ROOT_DIR
from hcl.entity import DatasetManager, TextDatasets
from hcl.entity.clusterer import Clusterer
from hcl.entity.dataset import DatasetFactory
from hcl.experiments.methods import METHODS
from hcl.utils import evaluate
from hcl.utils.config import get_config
from hcl.utils.data import random_subset
from hcl.utils.parallel import gather, task

METHOD_FACTORIES = dict(METHODS)


@task
def run(
    method_name: str,
    dataset_id: str,
    dataset_name: str,
    model_name: str,
    min_cluster_size: int,
    *,
    split: int | None = None,
    n_clusters: int | None = None,
) -> dict[str, Any]:
    method_factory = METHOD_FACTORIES[method_name]
    dataset = DatasetManager(model_name).get(DatasetFactory.from_string(dataset_id), skip_cache=True)
    x, y = dataset.embeddings, dataset.labels
    if split is not None:
        x, y = random_subset(x, y, n=min(1000, x.shape[0]), seed=split)
        x, y = x.copy(), y.copy()  # Full array is memmap'd; copy to keep only the subset in RAM for the worker process
    if n_clusters is None:
        n_clusters = int(np.unique(y).shape[0])
    if method_name == "HDBSCAN":
        clusterer: Clusterer = method_factory(-1, min_cluster_size=min_cluster_size)
    else:
        clusterer: Clusterer = method_factory(n_clusters)
    if isinstance(clusterer, AgglomerativeClustering):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.linkage}"
    elif isinstance(clusterer, VonMisesFisherMixture):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.posterior_type}"
    elif isinstance(clusterer, Pipeline):
        clusterer_name = "_".join([name for name, _ in clusterer.steps])
    else:
        clusterer_name = clusterer.__class__.__name__
    result = evaluate(clusterer, dataset_name, x, y, clusterer_name)
    return {**result.to_dict(), "model_name": model_name, "split": split}


def embedding_models_over_k():
    config = get_config()
    all_results = []
    jobs = []
    hdbscan_params = pd.read_csv(ROOT_DIR / "results" / "data" / "hdbscan_params.csv")

    datasets = [
        (TextDatasets.DBPEDIA_14, "DBPedia"),
        (TextDatasets.BUILT_BENCH_CLUSTERING_P2P, "BuiltBenchP2P"),
        (TextDatasets.CLUSTREC_COVID, "ClusTREC-Covid"),
        (TextDatasets.WIKICITIES, "WikiCities"),
    ]
    for i, (dataset, dataset_name) in enumerate(datasets):
        row = hdbscan_params[hdbscan_params["dataset"] == dataset_name].iloc[0]
        min_cluster_size = int(row["min_cluster_size"])

        for model_name in config["models"]:
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
                            min_cluster_size=min_cluster_size,
                        )
                    )

        # Use the default loky (process) backend so workers are not limited by the GIL.
        # With mmap_mode='r' in TextDataset.embeddings the OS page cache is shared
        # across worker processes, so the embedding arrays are not duplicated in RAM.
        if i % 5 == 0:
            results = gather(jobs, show_progress=True)
            all_results.extend(results)
            jobs = []
    df = pd.DataFrame(all_results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_embedding_models_k.csv", index=False)


def embedding_models():
    config = get_config()
    all_results = []
    jobs = []
    splits = 25
    hdbscan_params = pd.read_csv(ROOT_DIR / "results" / "data" / "hdbscan_params.csv")

    for _i, dataset_name in enumerate(config["datasets"]):
        row = hdbscan_params[hdbscan_params["dataset"] == dataset_name].iloc[0]
        min_cluster_size = int(row["min_cluster_size"])

        for model_name in config["models"]:
            for split in range(splits):
                for method_name, _ in METHODS:
                    jobs.append(
                        run(
                            method_name=method_name,
                            dataset_id=dataset_name,
                            dataset_name=dataset_name,
                            model_name=model_name,
                            split=split,
                            min_cluster_size=min_cluster_size,
                        )
                    )

        # if i % 5 == 0:
        #     results = gather(jobs, show_progress=True)
        #     all_results.extend(results)
        #     jobs = []

    results = gather(jobs, show_progress=True)
    all_results.extend(results)
    df = pd.DataFrame(all_results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_embedding_models.csv", index=False)


if __name__ == "__main__":
    embedding_models()
    # embedding_models_over_k()
