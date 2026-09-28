import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import Pipeline

from hcl.baselines import VonMisesFisherMixture
from hcl.definitions import ROOT_DIR
from hcl.entity.clusterer import Clusterer
from hcl.entity.dataset import DatasetFactory, DatasetManager
from hcl.experiments.methods import METHODS
from hcl.utils import evaluate
from hcl.utils.config import get_config
from hcl.utils.data import prepare_data
from hcl.utils.parallel import gather, task


@task
def run(clusterer: Clusterer, x: np.ndarray, y: np.ndarray, dataset_name: str, split: int) -> dict[str, Any]:
    if isinstance(clusterer, AgglomerativeClustering):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.linkage}"
    elif isinstance(clusterer, VonMisesFisherMixture):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.posterior_type}"
    elif isinstance(clusterer, Pipeline):
        clusterer_name = "_".join([name for name, _ in clusterer.steps])
    else:
        clusterer_name = clusterer.__class__.__name__
    result = evaluate(clusterer, dataset_name, x, y, clusterer_name)
    return {**result.to_dict(), "split": split}


def main():
    config = get_config()
    model_name = "intfloat/multilingual-e5-large"
    n_splits = 25

    hdbscan_params = pd.read_csv(ROOT_DIR / "results" / "data" / "hdbscan_params.csv")

    jobs = []

    for dataset_name in config["datasets"]:
        dataset = DatasetManager(model_name).get(DatasetFactory.from_string(dataset_name))
        row = hdbscan_params[hdbscan_params["dataset"] == dataset_name].iloc[0]
        min_cluster_size = int(row["min_cluster_size"])

        for split in range(n_splits):
            x, y = prepare_data(dataset.embeddings, dataset.labels, n=min(1000, dataset.labels.shape[0]), seed=split)
            k = np.unique(y).shape[0]
            for method_name, method_factory in METHODS:
                if method_name == "HDBSCAN":
                    method = method_factory(-1, min_cluster_size=min_cluster_size)
                else:
                    method = method_factory(k)
                jobs.append(run(method, x, y, dataset_name, split))

    results = gather(jobs, show_progress=True)
    hdbscan_params = pd.DataFrame(results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    hdbscan_params.to_csv(ROOT_DIR / "results" / "data" / "explore_datasets.csv", index=False)


if __name__ == "__main__":
    main()
