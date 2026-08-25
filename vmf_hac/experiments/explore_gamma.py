import os
from typing import Any

import numpy as np
import pandas as pd

from vmf_hac import VmfHAC
from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity.dataset import DatasetFactory, DatasetManager
from vmf_hac.utils import evaluate, gather, get_config, task
from vmf_hac.utils.data import prepare_data


@task
def run(x: np.ndarray, y: np.ndarray, gamma: float, dataset_name: str) -> dict[str, Any]:
    n_clusters = np.unique(y).shape[0]
    result = evaluate(VmfHAC(n_clusters=n_clusters, gamma=gamma, should_normalize=False), dataset_name, x, y)
    return result.to_dict()


def main():
    config = get_config()
    # gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gammas = np.linspace(0.001, 1.0, 100).tolist()
    jobs = []
    for dataset_name in config["datasets"]:
        dataset = DatasetManager("intfloat/multilingual-e5-large").get(DatasetFactory.from_string(dataset_name))
        x, y = prepare_data(dataset.embeddings, dataset.labels, n=1000)
        for gamma in gammas:
            jobs.append(run(x, y, gamma=gamma, dataset_name=dataset_name))

    results = gather(jobs, show_progress=True)
    df = pd.DataFrame(results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_gamma.csv", index=False)


if __name__ == "__main__":
    main()
