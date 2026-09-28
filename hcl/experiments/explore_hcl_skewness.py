import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew

from hcl.core import HclHAC
from hcl.definitions import ROOT_DIR
from hcl.entity.dataset import DatasetFactory, DatasetManager
from hcl.utils import gather, get_config, task
from hcl.utils.data import prepare_data


@task
def run(
    gamma: float,
    x: np.ndarray,
    dataset_name: str,
    split: int,
) -> list[dict[str, Any]]:
    model = HclHAC(n_clusters=1, should_normalize=False, gamma=gamma).fit(x)
    results = []
    for n_clusters in range(1, x.shape[0]):
        model.n_clusters = n_clusters
        hcl_labels = model.predict(x)

        hcl_cluster_sizes = np.unique_counts(hcl_labels).counts
        hcl_skew = skew(hcl_cluster_sizes)
        results.append(
            {
                "gamma": gamma,
                "hcl_cluster_skew": hcl_skew,
                "dataset_name": dataset_name,
                "split": split,
            }
        )
    return results


def main():
    config = get_config()
    # gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gammas = np.linspace(0.001, 1.0, 100).tolist()
    n_splits = 5
    jobs = []
    for dataset_name in config["datasets"]:
        dataset = DatasetManager("intfloat/multilingual-e5-large").get(DatasetFactory.from_string(dataset_name))
        for split in range(n_splits):
            x, _ = prepare_data(dataset.embeddings, dataset.labels, n=1000, seed=split)

            for gamma in gammas:
                jobs.append(
                    run(
                        gamma=gamma,
                        x=x,
                        dataset_name=dataset_name,
                        split=split,
                    )
                )

    results = gather(jobs, show_progress=True)
    all_results = []
    for result in results:
        all_results.extend(result)
    df = pd.DataFrame(all_results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_hcl_skewness.csv", index=False)


if __name__ == "__main__":
    main()
