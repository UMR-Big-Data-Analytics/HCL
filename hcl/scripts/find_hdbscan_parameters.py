import json

import optuna
import pandas as pd
from sklearn.cluster import HDBSCAN
from umap import UMAP

from hcl.definitions import ROOT_DIR
from hcl.entity import DatasetFactory, DatasetManager, TextDatasets
from hcl.utils import evaluate, random_subset


def find_hdbscan_params(dataset_manager: DatasetManager, dataset_id: TextDatasets):
    dataset = dataset_manager.get(dataset_id)
    # Held-out seed: splits 0-4 are used for evaluation, so tuning must not reuse one of them.
    x, y = random_subset(dataset.embeddings, dataset.labels, min(1000, dataset.labels.shape[0]), seed=1000)
    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    x_red = umap.fit_transform(x)

    def search(trial):
        min_cluster_size = trial.suggest_int("min_cluster_size", 2, 100)
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size)
        res = evaluate(hdbscan, str(dataset_id), x_red, y)
        cluster_diff = res.n_clusters - res.n_true_clusters
        if cluster_diff < 0:
            cluster_diff = cluster_diff * -1
        cluster_diff = cluster_diff / res.n_true_clusters
        # Minimize noise, make computed clusters as close as possible to true number of clusters
        return res.v_measure * (1 - res.noise_fraction) * (1 - cluster_diff)

    study = optuna.create_study(direction="maximize")
    study.optimize(search, n_trials=100, timeout=60)
    return study.best_params


def main():
    with open(ROOT_DIR / "config.json") as f:
        config = json.load(f)
    results = []
    dataset_manager = DatasetManager("intfloat/multilingual-e5-large")
    for dataset_name in config["datasets"]:
        dataset_id = DatasetFactory.from_string(dataset_name)
        best_params = find_hdbscan_params(dataset_manager, dataset_id)
        results.append({"dataset": dataset_name, "min_cluster_size": best_params["min_cluster_size"]})
    df = pd.DataFrame(results)
    df.to_csv(ROOT_DIR / "results" / "data" / "hdbscan_params.csv", index=False)


if __name__ == "__main__":
    main()
