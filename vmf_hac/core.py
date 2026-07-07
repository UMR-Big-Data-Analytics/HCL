from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator, ClusterMixin

from vmf_hac.entity.cluster import Cluster
from vmf_hac.functions.norm import normalize
from vmf_hac.functions.vmf import spherical_ward_linkage, vmf_linkage


class VmfHAC(BaseEstimator, ClusterMixin):
    def __init__(
        self, n_clusters: int, gamma: float, linkage_fn: Literal["vmf", "spherical_ward"] = "vmf", should_normalize=True
    ):
        self.labels_ = None
        self.linkage_matrix_ = None
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.linkage_fn = linkage_fn
        self.should_normalize = should_normalize

    def fit(self, X: np.ndarray, y=None):
        self.linkage_matrix_ = vmf_hac(X, self.gamma, self.linkage_fn, self.should_normalize)
        return self

    def fit_predict(self, X: np.ndarray, y=None, **kwargs) -> np.ndarray:
        if self.linkage_matrix_ is None:
            self.fit(X, y)
        return self.predict(X)

    def predict(self, _: np.ndarray) -> np.ndarray:
        self.labels_ = self.predict_cluster(self.n_clusters)
        return self.labels_

    def predict_cluster(self, n_clusters: int) -> np.ndarray:
        return fcluster(self.linkage_matrix_, t=n_clusters, criterion="maxclust")  # ty:ignore[no-matching-overload]


def vmf_hac(X: np.ndarray, gamma: float, linkage_fn="vmf", should_normalize=True):
    min_cost = 9999999
    n, _ = X.shape
    if should_normalize:
        X = normalize(X)
    clusters = {i: Cluster.singleton(i, X[i]) for i in range(n)}
    active = list(range(n))
    next_cluster_id = n
    linkage = np.zeros((n - 1, 4))

    cost_matrix = np.full((2 * n, 2 * n), np.inf, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            S_ab = clusters[i].S + clusters[j].S
            R_ab = float(np.linalg.norm(S_ab))

            if linkage_fn == "vmf":
                cost = vmf_linkage(
                    clusters[i].N,
                    clusters[i].R,
                    clusters[j].N,
                    clusters[j].R,
                    clusters[i].N + clusters[j].N,
                    R_ab,
                    gamma,
                )
            elif linkage_fn == "spherical_ward":
                cost = spherical_ward_linkage(
                    clusters[i].S,
                    clusters[j].S,
                )
            else:
                raise ValueError(f"Unknown linkage function: {linkage_fn}")
            if cost < min_cost:
                min_cost = cost
            cost_matrix[i, j] = cost
            cost_matrix[j, i] = cost

    for merge_step in range(n - 1):
        active_arr = np.array(active)

        sub = cost_matrix[np.ix_(active_arr, active_arr)]
        idx = np.argmin(sub)
        a_idx, b_idx = np.unravel_index(idx, sub.shape)

        i, j = active_arr[a_idx], active_arr[b_idx]
        if j < i:
            i, j = j, i

        cost = cost_matrix[i, j]
        A, B = clusters[i], clusters[j]
        merged = Cluster.merge_clusters(A, B, next_cluster_id)
        clusters[next_cluster_id] = merged

        linkage[merge_step] = [i, j, cost, merged.N]

        active.remove(i)
        active.remove(j)

        for k in active:
            S_ab = merged.S + clusters[k].S
            R_ab = float(np.linalg.norm(S_ab))

            if linkage_fn == "vmf":
                c = vmf_linkage(merged.N, merged.R, clusters[k].N, clusters[k].R, merged.N + clusters[k].N, R_ab, gamma)
            elif linkage_fn == "spherical_ward":
                c = spherical_ward_linkage(
                    merged.S,
                    clusters[k].S,
                )
            else:
                raise ValueError(f"Unknown linkage function: {linkage_fn}")
            if c < min_cost:
                min_cost = c
            cost_matrix[next_cluster_id, k] = c
            cost_matrix[k, next_cluster_id] = c

        active.append(next_cluster_id)
        next_cluster_id += 1

    # print("Min cost:", min_cost)

    # Sometimes likelihood ratios yield tiny negative numbers. Fix them
    min_value = np.min(linkage[:, 2])
    if min_value < 0:
        linkage[:, 2] = linkage[:, 2] - min_value

    return linkage
