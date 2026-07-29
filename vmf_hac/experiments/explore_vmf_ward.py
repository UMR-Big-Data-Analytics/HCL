import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, skew, spearmanr

from vmf_hac import VmfHAC
from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity.dataset import DatasetFactory, DatasetManager
from vmf_hac.utils import gather, get_config, task
from vmf_hac.utils.data import prepare_data


@task
def run(
    n_clusters: int,
    gamma: float,
    x: np.ndarray,
    dataset_name: str,
    ward_cop,
    ward_skew: float,
    split: int,
) -> dict[str, Any]:
    model = VmfHAC(n_clusters=n_clusters, should_normalize=False, gamma=gamma).fit(x)
    z_vmf = model.linkage_matrix_
    vmf_labels = model.predict(x)

    vmf_cop = cophenet(z_vmf)
    pearson_corr = pearsonr(ward_cop, vmf_cop)[0]
    spearman_corr = spearmanr(ward_cop, vmf_cop)[0]

    vmf_cluster_sizes = np.unique_counts(vmf_labels).counts
    vmf_skew = skew(vmf_cluster_sizes)

    return {
        "gamma": gamma,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "ward_cluster_skew": ward_skew,
        "vmf_cluster_skew": vmf_skew,
        "dataset_name": dataset_name,
        "split": split,
    }


def main():
    config = get_config()
    # gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gammas = np.linspace(0.001, 1.0, 100).tolist()
    n_splits = 5
    jobs = []
    for dataset_name in config["datasets"]:
        dataset = DatasetManager("intfloat/multilingual-e5-large").get(DatasetFactory.from_string(dataset_name))
        for split in range(n_splits):
            x, y = prepare_data(dataset.embeddings, dataset.labels, n=1000, seed=split)
            n_clusters = np.unique(y).shape[0]

            # Compute ward; this time with scipy to get the linkage matrix
            d = pdist(x, metric="euclidean")
            z_ward = linkage(d, method="ward")

            cop_ward = cophenet(z_ward)
            ward_labels = fcluster(z_ward, t=n_clusters, criterion="maxclust")
            ward_cluster_sizes = np.unique_counts(ward_labels).counts
            ward_cluster_skew = skew(ward_cluster_sizes)

            for gamma in gammas:
                jobs.append(
                    run(
                        n_clusters=n_clusters,
                        gamma=gamma,
                        x=x,
                        dataset_name=dataset_name,
                        ward_cop=cop_ward,
                        ward_skew=ward_cluster_skew,
                        split=split,
                    )
                )

    results = gather(jobs, show_progress=True)
    df = pd.DataFrame(results)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_vmf_ward.csv", index=False)


if __name__ == "__main__":
    main()
