import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from vmf_hac.baselines import VonMisesFisherMixture
from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity.clusterer import Clusterer
from vmf_hac.entity.dataset import DatasetFactory, DatasetManager
from vmf_hac.experiments.methods import METHODS
from vmf_hac.utils import evaluate
from vmf_hac.utils.config import get_config
from vmf_hac.utils.data import prepare_data
from vmf_hac.utils.parallel import gather, task


@task
def run(clusterer: Clusterer, x: np.ndarray, y: np.ndarray, dataset_name: str, split: int) -> dict[str, Any]:
    if isinstance(clusterer, AgglomerativeClustering):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.linkage}"
    elif isinstance(clusterer, VonMisesFisherMixture):
        clusterer_name = f"{clusterer.__class__.__name__}_{clusterer.posterior_type}"
    else:
        clusterer_name = clusterer.__class__.__name__
    result = evaluate(clusterer, dataset_name, x, y, clusterer_name)
    return {**result.to_dict(), "split": split}


def main():
    config = get_config()
    model_name = "intfloat/multilingual-e5-large"
    n_splits = 25

    jobs = []

    for dataset_name in config["datasets"]:
        dataset = DatasetManager(model_name).get(DatasetFactory.from_string(dataset_name))
        for split in range(n_splits):
            x, y = prepare_data(dataset.embeddings, dataset.labels, n=1000, seed=split)
            k = np.unique(y).shape[0]
            for _, method_factory in METHODS:
                method = method_factory(k)
                jobs.append(run(method, x, y, dataset_name, split))

    results = gather(jobs, show_progress=True)
    df = pd.DataFrame(results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_datasets.csv", index=False)


if __name__ == "__main__":
    main()
