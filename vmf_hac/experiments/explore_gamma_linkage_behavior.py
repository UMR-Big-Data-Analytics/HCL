import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, ward
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, v_measure_score

from vmf_hac import VmfHAC
from vmf_hac.definitions import ROOT_DIR
from vmf_hac.entity import DatasetManager
from vmf_hac.entity.dataset import TextDatasets
from vmf_hac.utils.data import random_subset
from vmf_hac.utils.parallel import gather, task


@task
def compute_vmf_for_gamma(dataset_label: str, x: np.ndarray, gamma: float) -> tuple[str, float, VmfHAC]:
    return dataset_label, gamma, VmfHAC(-1, gamma=gamma).fit(x)


def compute_tree_trajectory(
    x: np.ndarray,
    y_true: np.ndarray,
    dataset_name: str,
    gammas: list[float],
    max_k: int,
    vmfs: dict[float, VmfHAC],
) -> pd.DataFrame:
    n_samples = x.shape[0]
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    z_ward = ward(pdist(x_norm))
    k_values = np.linspace(2, min(max_k, n_samples // 2), 50).astype(int)[::-1]
    n_true_clusters = np.unique(y_true).shape[0]

    rows = []
    for k in k_values:
        labels_ward = fcluster(z_ward, t=k, criterion="maxclust")
        ari_ward = adjusted_rand_score(y_true, labels_ward)
        v_measure_ward = v_measure_score(y_true, labels_ward)
        for gamma in gammas:
            labels_vmf = vmfs[gamma].predict_cluster(k)
            rows.append(
                {
                    "gamma": gamma,
                    "k": k,
                    "ari_vmf": adjusted_rand_score(y_true, labels_vmf),
                    "v_measure_vmf": v_measure_score(y_true, labels_vmf),
                    "ari_ward": ari_ward,
                    "v_measure_ward": v_measure_ward,
                    "dataset_name": dataset_name,
                    "n_true_clusters": n_true_clusters,
                }
            )
    return pd.DataFrame(rows)


def main():
    model_name = "intfloat/multilingual-e5-large"
    gammas = [0.025, 0.05, 0.075, 0.1]
    datasets = [
        (TextDatasets.DBPEDIA_14, "DBPedia"),
        (TextDatasets.BANKING77, "Banking77"),
        (TextDatasets.TWENTY_NEWSGROUPS, "20Newsgroups"),
        (TextDatasets.WIKICITIES, "WikiCities"),
    ]

    manager = DatasetManager(model_name)
    dataset_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dataset_key, dataset_label in datasets:
        dataset = manager.get(dataset_key)
        x, y = random_subset(dataset.embeddings, dataset.labels, n=1000, seed=42)
        dataset_data[dataset_label] = (x, y)

    vmf_results = gather(
        [
            compute_vmf_for_gamma(dataset_label, dataset_data[dataset_label][0], gamma)
            for _, dataset_label in datasets
            for gamma in gammas
        ],
        show_progress=True,
    )
    vmfs_by_dataset: dict[str, dict[float, VmfHAC]] = {dataset_label: {} for _, dataset_label in datasets}
    for dataset_label, gamma, vmf_impl in vmf_results:
        vmfs_by_dataset[dataset_label][gamma] = vmf_impl

    dfs = []
    for _, dataset_label in datasets:
        x, y = dataset_data[dataset_label]
        max_k = int(np.unique(y).shape[0] * 1.5)
        dfs.append(
            compute_tree_trajectory(
                x=x,
                y_true=y,
                dataset_name=dataset_label,
                gammas=gammas,
                max_k=max_k,
                vmfs=vmfs_by_dataset[dataset_label],
            )
        )

    df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    os.makedirs(ROOT_DIR / "results" / "data", exist_ok=True)
    df.to_csv(ROOT_DIR / "results" / "data" / "explore_gamma_linkage_behavior.csv", index=False)


if __name__ == "__main__":
    main()
