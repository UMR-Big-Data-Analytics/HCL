import time
from warnings import warn

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


def _labels_noise_as_singletons(y_pred: np.ndarray) -> np.ndarray:
    """Give every noise point its own cluster, so noise is scored rather than discarded."""
    labels = y_pred.copy()
    noise_idx = np.flatnonzero(labels == -1)
    if noise_idx.size:
        labels[noise_idx] = labels.max() + 1 + np.arange(noise_idx.size)
    return labels


def _labels_noise_to_nearest(x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Assign every noise point to the cosine-nearest non-noise cluster centroid."""
    noise_mask = y_pred == -1
    if not noise_mask.any():
        return y_pred.copy()
    if noise_mask.all():
        return _labels_noise_as_singletons(y_pred)

    cluster_ids = np.unique(y_pred[~noise_mask])
    centroids = np.stack([x[y_pred == cluster_id].mean(axis=0) for cluster_id in cluster_ids])

    def _unit(a: np.ndarray) -> np.ndarray:
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

    similarity = _unit(x[noise_mask]) @ _unit(centroids).T
    labels = y_pred.copy()
    labels[noise_mask] = cluster_ids[similarity.argmax(axis=1)]
    return labels


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

    noise_mask = y_pred == -1
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

    # Noise-inclusive variants: v_measure above only scores the points the clusterer chose to
    # label, which flatters methods that abstain. These score all n samples instead.
    # v_measure_singleton is an optimistic bound only - singletons are perfectly homogeneous, so
    # V-measure actually *rewards* abstention. Use v_measure_nearest for fair comparison.
    labels_nearest = _labels_noise_to_nearest(x, y_pred)
    v_measure_singleton = v_measure_score(y, _labels_noise_as_singletons(y_pred))
    v_measure_nearest = v_measure_score(y, labels_nearest)
    # V-measure is not corrected for chance and rewards over-clustering, so a method free to pick
    # its own cluster count can inflate it. ami/ari_nearest are chance-corrected and do not.
    ami_nearest = adjusted_mutual_info_score(y, labels_nearest)
    ari_nearest = adjusted_rand_score(y, labels_nearest)

    n_non_noise_samples = labels_pred_non_noise.shape[0]
    n_pred_clusters = np.unique(labels_pred_non_noise).shape[0]

    has_valid_structure = 2 <= n_pred_clusters < n_non_noise_samples
    if not has_valid_structure:
        warn(
            (
                "Internal clustering metrics undefined for "
                f"{dataset_name}/{clusterer_name or clusterer.__class__.__name__}: "
                f"n_clusters={n_pred_clusters}, n_samples={n_non_noise_samples}. "
                "Setting silhouette/calinski_harabasz/davies_bouldin to NaN."
            ),
            stacklevel=2,
        )
        silhouette = float("nan")
        calinski_harabasz = float("nan")
        davies_bouldin = float("nan")
    else:
        silhouette = silhouette_score(embeddings_non_noise, labels_pred_non_noise)
        calinski_harabasz = calinski_harabasz_score(embeddings_non_noise, labels_pred_non_noise)
        davies_bouldin = davies_bouldin_score(embeddings_non_noise, labels_pred_non_noise)

    noise_fraction = noise_mask.sum() / y_pred.shape[0]
    n_clusters = n_pred_clusters
    n_true_clusters = np.unique(y).shape[0]

    return ExperimentResult(
        clusterer_name=clusterer_name if clusterer_name is not None else clusterer.__class__.__name__,
        dataset_name=dataset_name,
        dim=dim,
        nmi=nmi,
        ari=ari,
        v_measure=v_measure,
        v_measure_singleton=v_measure_singleton,
        v_measure_nearest=v_measure_nearest,
        ami_nearest=ami_nearest,
        ari_nearest=ari_nearest,
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
