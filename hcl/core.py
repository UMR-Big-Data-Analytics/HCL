import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator, ClusterMixin

from hcl.functions import normalize


class HclHAC(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_clusters: int,
        gamma: float | tuple[float, float],
        progressive: bool = False,
        should_normalize=True,
    ):
        self.labels_ = None
        self.linkage_matrix_ = None
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.progressive = progressive
        if self.progressive and not isinstance(self.gamma, tuple):
            raise ValueError("When progressive is True, gamma must be a tuple of (start, end) values.")
        self.should_normalize = should_normalize

    def fit(self, X: np.ndarray, y=None):
        if y is not None:
            raise ValueError("HlcHAC is an unsupervised clustering algorithm and does not support fitting with labels.")
        # float16 can result in numerical overflows when using numpy.linalg.norm with high dimensional data
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        self.linkage_matrix_ = hcl_hac(X, self.gamma, self.progressive, self.should_normalize)
        return self

    def fit_predict(self, X: np.ndarray, y=None, **kwargs) -> np.ndarray:
        if self.linkage_matrix_ is None:
            self.fit(X, y)
        return self.predict(X)

    def predict(self, _: np.ndarray) -> np.ndarray:
        self.labels_ = self.predict_cluster(self.n_clusters)
        return self.labels_

    def predict_cluster(self, n_clusters: int) -> np.ndarray:
        # vMF merge costs are not monotone (inversions, especially in progressive mode where the
        # cost shrinks as gamma grows), so cutting by height with "maxclust" often cannot hit
        # exactly n_clusters and neighbouring k collapse onto the same partition. Cutting by merge
        # order instead (undo the last n_clusters - 1 merges) always yields exactly n_clusters.
        assert self.linkage_matrix_ is not None
        merge_order = self.linkage_matrix_.copy()
        merge_order[:, 2] = np.arange(merge_order.shape[0], dtype=np.float64)
        return fcluster(merge_order, t=max(n_clusters, 1), criterion="maxclust")


def _vmf_scores(N: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    # Vectorised vmf_score; R may exceed N by rounding errors, so clip before the log.
    return N * np.log(1.0 - np.minimum(R / N, 1.0) + gamma)


def _pair_costs(
    rows: np.ndarray,
    cols: np.ndarray,
    N: np.ndarray,
    G: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Linkage costs between the clusters in slots ``rows`` and ``cols``.

    ``G`` is the Gram matrix of the cluster resultant vectors, i.e. ``G[a, b] = S_a . S_b``,
    so ``||S_a + S_b||^2 = G[a, a] + G[b, b] + 2 G[a, b]``.
    """
    # squared resultant lengths ||S_a||^2 of every slot (clip rounding noise below zero)
    sq_r = np.maximum(np.diag(G), 0.0)
    # R_A as a column and R_B as a row vector, so broadcasting yields a (rows x cols) grid
    R_rows = np.sqrt(sq_r[rows])[:, None]
    R_cols = np.sqrt(sq_r[cols])[None, :]
    # R_AB = ||S_a + S_b|| for every pair, computed from the Gram matrix without touching X
    R_ab = np.sqrt(np.maximum(sq_r[rows][:, None] + sq_r[cols][None, :] + 2.0 * G[np.ix_(rows, cols)], 0.0))

    N_rows = N[rows][:, None]
    N_cols = N[cols][None, :]
    # same formula as vmf_linkage: score(A u B) - score(A) - score(B), for all pairs at once
    return (
        _vmf_scores(N_rows + N_cols, R_ab, gamma)
        - _vmf_scores(N_rows, R_rows, gamma)
        - _vmf_scores(N_cols, R_cols, gamma)
    )


def hcl_hac(
    X: np.ndarray,
    gamma: float | tuple[float, float],
    progressive: bool = False,
    should_normalize=True,
):
    """Agglomerative clustering with the vMF (or spherical Ward) linkage.

    With ``progressive=True``, ``gamma`` is a ``(start, end)`` tuple and gamma is increased
    linearly from ``start`` (first merge) to ``end`` (last merge). Since the linkage cost depends
    on gamma, all pairwise costs among the active clusters are re-evaluated at the current gamma
    before every merge, so that the merge decision never compares costs computed with different
    gammas.
    """
    n, _ = X.shape
    if should_normalize:
        X = normalize(X)

    # gammas[t] is the gamma used to score the candidates for merge step t
    if progressive:
        if not isinstance(gamma, tuple) or len(gamma) != 2:
            raise ValueError("When progressive is True, gamma must be a tuple of (start, end) values.")
        gammas = np.linspace(gamma[0], gamma[1], max(n - 1, 1))
    else:
        if isinstance(gamma, tuple):
            raise ValueError("gamma must be a scalar when progressive is False.")
        gammas = np.full(max(n - 1, 1), float(gamma))
    recompute = progressive

    # State is stored per "slot" (row index 0..n-1) instead of per cluster id. When two clusters
    # merge, the result reuses the slot of one child and the other slot is deactivated, so all
    # matrices stay n x n even though scipy's linkage format numbers clusters from 0 to 2n - 2.
    # G[a, b] = S_a . S_b where S_a is the sum of the (unit) vectors in slot a. For singletons
    # this is just the cosine similarity matrix.
    G = X @ X.T
    # N[a] = number of points in the cluster held by slot a
    N = np.ones(n, dtype=np.float64)
    # slot_ids[a] = scipy cluster id of the cluster currently held by slot a
    slot_ids = np.arange(n)
    # active[a] = whether slot a still holds a cluster that can be merged
    active = np.ones(n, dtype=bool)

    all_slots = np.arange(n)
    # cost_matrix[a, b] = linkage cost of merging slots a and b (inf = never merge)
    cost_matrix = _pair_costs(all_slots, all_slots, N, G, gammas[0])
    np.fill_diagonal(cost_matrix, np.inf)

    linkage = np.zeros((n - 1, 4))
    for merge_step in range(n - 1):
        active_slots = np.flatnonzero(active)
        if recompute and merge_step > 0:
            # Progressive: gamma changed since the last step, so every cached cost is stale.
            # Rescore all active pairs at the current gamma so the argmin compares like with like.
            sub = _pair_costs(active_slots, active_slots, N, G, gammas[merge_step])
            np.fill_diagonal(sub, np.inf)
            cost_matrix[np.ix_(active_slots, active_slots)] = sub
        else:
            # Fixed gamma: cached costs are still valid, only the merged row was updated below.
            sub = cost_matrix[np.ix_(active_slots, active_slots)]

        # cheapest pair among active clusters; indices are local to sub, map them back to slots
        a_idx, b_idx = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = active_slots[a_idx], active_slots[b_idx]
        cost = sub[a_idx, b_idx]

        # record the merge in scipy format: [child id, child id, height, size of new cluster]
        id_a, id_b = slot_ids[a], slot_ids[b]
        N_ab = N[a] + N[b]
        linkage[merge_step] = [min(id_a, id_b), max(id_a, id_b), cost, N_ab]

        # Merge b into slot a. Since S_ab = S_a + S_b, S_ab . S_k = G[a, k] + G[b, k], so the
        # Gram row and column of a are updated by adding those of b. Doing the row first and
        # then the column also yields the correct diagonal G[a, a] = ||S_a + S_b||^2.
        G[a, :] += G[b, :]
        G[:, a] += G[:, b]
        N[a] = N_ab
        # the new cluster gets the next scipy id (n, n + 1, ...)
        slot_ids[a] = n + merge_step
        # retire slot b so it is never picked again
        active[b] = False
        cost_matrix[b, :] = np.inf
        cost_matrix[:, b] = np.inf

        # Fixed gamma: only costs involving the new cluster changed, so rescore just row/column a
        # against all other active slots. (Progressive mode rescores everything next step.)
        others = np.flatnonzero(active & (all_slots != a))
        if not recompute and others.size:
            gamma_next = gammas[min(merge_step + 1, n - 2)]
            c = _pair_costs(np.array([a]), others, N, G, gamma_next)[0]
            cost_matrix[a, others] = c
            cost_matrix[others, a] = c

    # Sometimes likelihood ratios yield tiny negative numbers. Fix them
    min_value = np.min(linkage[:, 2])
    if min_value < 0:
        linkage[:, 2] = linkage[:, 2] - min_value

    return linkage
