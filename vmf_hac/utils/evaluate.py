import time

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

from vmf_hac.entity.clusterer import Clusterer
from vmf_hac.entity.experiment_result import ExperimentResult


def evaluate(
    clusterer: Clusterer,
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    clusterer_name: str | None = None,
) -> ExperimentResult:
    start = time.perf_counter()
    y_pred = clusterer.fit_predict(x)
    end = time.perf_counter()

    dim = x.shape[1]
    running_time_s = end - start

    noise_mask = y == -1
    labels_pred_non_noise = y_pred[~noise_mask]
    labels_true_non_noise = y[~noise_mask]

    embeddings_non_noise = x[~noise_mask]

    nmi = normalized_mutual_info_score(labels_true_non_noise, labels_pred_non_noise)
    ari = adjusted_rand_score(labels_true_non_noise, labels_pred_non_noise)
    ami = adjusted_mutual_info_score(labels_true_non_noise, labels_pred_non_noise)
    homogeneity = homogeneity_score(labels_true_non_noise, labels_pred_non_noise)
    completeness = completeness_score(labels_true_non_noise, labels_pred_non_noise)
    v_measure = v_measure_score(labels_true_non_noise, labels_pred_non_noise)
    fowlkes_mallows = fowlkes_mallows_score(labels_true_non_noise, labels_pred_non_noise)

    silhouette = silhouette_score(embeddings_non_noise, labels_pred_non_noise)
    calinski_harabasz = calinski_harabasz_score(embeddings_non_noise, labels_pred_non_noise)
    davies_bouldin = davies_bouldin_score(embeddings_non_noise, labels_pred_non_noise)

    noise_fraction = float(noise_mask.sum()) / y_pred.shape[0]
    n_clusters = np.unique(labels_pred_non_noise).shape[0]
    n_true_clusters = np.unique(y).shape[0]

    return ExperimentResult(
        clusterer_name=clusterer_name if clusterer_name is not None else clusterer.__class__.__name__,
        dataset_name=dataset_name,
        dim=dim,
        nmi=nmi,
        ari=ari,
        v_measure=v_measure,
        noise_fraction=noise_fraction,
        homogeneity=homogeneity,
        completeness=completeness,
        ami=ami,
        fowlkes_mallows=fowlkes_mallows,
        silhouette=silhouette,
        calinski_harabasz=calinski_harabasz,
        davies_bouldin=davies_bouldin,
        n_clusters=n_clusters,
        n_true_clusters=n_true_clusters,
        running_time_s=running_time_s,
        params=clusterer.get_params(),
    )
