from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.special import ive, logsumexp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted


Algorithm = Literal["EMb", "CEMb"]


class DBMOVMF(BaseEstimator, ClusterMixin):
    """
    Model-Based Co-clustering with Mixtures of von Mises-Fisher
    distributions (dbmovMFs).

    Based on:

        Salah, A., Rogovschi, N., & Nadif, M. (2016).
        Model-based Co-clustering for High Dimensional Sparse Data.
        AISTATS 2016.

    Parameters
    ----------
    n_clusters : int
        Number of row clusters and column clusters.

    algorithm : {"EMb", "CEMb"}, default="EMb"
        EMb performs soft expectation-maximization.

        CEMb performs classification EM, converting posterior row
        memberships to hard assignments before the M-step.

    n_init : int, default=10
        Number of random initializations.

    max_iter : int, default=100
        Maximum number of EM/CEM iterations per initialization.

    tol : float, default=1e-6
        Convergence tolerance.

        Convergence is tested using the maximum absolute change in the
        row membership matrix Z.

    random_state : int or numpy.random.Generator, optional
        Random seed or random number generator.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Final row-cluster assignments.

    column_labels_ : ndarray of shape (n_features,)
        Final column-cluster assignments.

    membership_ : ndarray of shape (n_samples, n_clusters)
        Final row membership matrix Z.

        For EMb, this contains posterior probabilities.

        For CEMb, this is a hard membership matrix.

    column_membership_ : ndarray of shape
        (n_features, n_clusters)
        Final column membership matrix W.

    alpha_ : ndarray of shape (n_clusters,)
        Mixture proportions.

    mu_ : ndarray of shape (n_clusters,)
        Non-zero diagonal centroid values mu_hh.

    kappa_ : ndarray of shape (n_clusters,)
        vMF concentration parameters.

    components_ : ndarray of shape
        (n_clusters, n_features)
        Full directional mean vectors mu_h^w.

    n_iter_ : int
        Number of iterations of the selected best run.

    lower_bound_ : float
        Observed-data log-likelihood of the final model.

    Notes
    -----
    The model assumes each row of X lies on the unit hypersphere.
    Input rows are therefore L2-normalized internally.

    The implementation supports signed data. The sign of mu_hh follows
    the derivation in the paper:

        mu_hh = sign(r_h^w) / sqrt(sum_j w_jh)

    where r_h^w is the aggregate directional statistic associated with
    co-cluster h.
    """

    def __init__(
        self,
        n_clusters: int,
        algorithm: Algorithm = "EMb",
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    # ================================================================
    # Validation
    # ================================================================

    def _validate_params(self) -> None:
        if not isinstance(self.n_clusters, int):
            raise TypeError(
                "n_clusters must be an integer."
            )

        if self.n_clusters < 1:
            raise ValueError(
                "n_clusters must be >= 1."
            )

        if self.algorithm not in ("EMb", "CEMb"):
            raise ValueError(
                "algorithm must be either 'EMb' or 'CEMb'."
            )

        if not isinstance(self.n_init, int) or self.n_init < 1:
            raise ValueError(
                "n_init must be a positive integer."
            )

        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError(
                "max_iter must be a positive integer."
            )

        if self.tol <= 0:
            raise ValueError(
                "tol must be > 0."
            )

        if self.verbose < 0:
            raise ValueError(
                "verbose must be >= 0."
            )

    @staticmethod
    def _check_X(X):
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=True,
        )

        if X.shape[0] == 0:
            raise ValueError(
                "X must contain at least one sample."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one feature."
            )

        return X

    # ================================================================
    # Normalization
    # ================================================================

    @staticmethod
    def _normalize_rows(X):
        """
        L2-normalize rows.

        Works for dense NumPy arrays and sparse matrices.
        """

        if sp.issparse(X):
            X = X.tocsr(copy=True)

            squared_norms = np.asarray(
                X.multiply(X).sum(axis=1)
            ).ravel()

            norms = np.sqrt(squared_norms)

            if np.any(norms == 0):
                raise ValueError(
                    "X contains one or more zero-norm rows."
                )

            inv_norms = 1.0 / norms

            return sp.diags(inv_norms) @ X

        norms = np.linalg.norm(
            X,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "X contains one or more zero-norm rows."
            )

        return X / norms

    # ================================================================
    # Random initialization
    # ================================================================

    @staticmethod
    def _initialize_labels(
        n: int,
        n_clusters: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Random hard assignment with guaranteed non-empty clusters.
        """

        if n_clusters > n:
            raise ValueError(
                "n_clusters cannot exceed the number of elements."
            )

        labels = np.empty(
            n,
            dtype=np.int64,
        )

        # Guarantee one element per cluster.
        labels[:n_clusters] = np.arange(
            n_clusters
        )

        # Assign remaining elements randomly.
        if n > n_clusters:
            labels[n_clusters:] = rng.integers(
                0,
                n_clusters,
                size=n - n_clusters,
            )

        rng.shuffle(labels)

        return labels

    @staticmethod
    def _labels_to_membership(
        labels: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        membership = np.zeros(
            (labels.shape[0], n_clusters),
            dtype=np.float64,
        )

        membership[
            np.arange(labels.shape[0]),
            labels,
        ] = 1.0

        return membership

    @staticmethod
    def _repair_empty_clusters(
        labels: np.ndarray,
        n_clusters: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Ensure every cluster has at least one assigned element.

        This is an implementation safeguard for degeneracies not
        explicitly addressed by the paper.
        """

        labels = labels.copy()

        counts = np.bincount(
            labels,
            minlength=n_clusters,
        )

        empty_clusters = np.flatnonzero(
            counts == 0
        )

        for empty_cluster in empty_clusters:

            donor_clusters = np.flatnonzero(
                counts > 1
            )

            if donor_clusters.size == 0:
                raise RuntimeError(
                    "Cannot repair empty clusters."
                )

            donor_cluster = rng.choice(
                donor_clusters
            )

            candidates = np.flatnonzero(
                labels == donor_cluster
            )

            moved_index = rng.choice(
                candidates
            )

            labels[moved_index] = empty_cluster

            counts[donor_cluster] -= 1
            counts[empty_cluster] += 1

        return labels

    # ================================================================
    # vMF normalization constant
    # ================================================================

    @staticmethod
    def _log_vmf_normalizer(
        kappa: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        """
        Compute log c_d(kappa), where

            c_d(kappa)
                =
                kappa^(d/2 - 1)
                /
                ((2*pi)^(d/2)
                I_{d/2 - 1}(kappa))

        Uses the exponentially scaled Bessel function:

            ive(v, x) = exp(-abs(x)) * iv(v, x)

        for numerical stability.
        """

        nu = n_features / 2.0 - 1.0

        kappa = np.asarray(
            kappa,
            dtype=np.float64,
        )

        kappa = np.maximum(
            kappa,
            np.finfo(np.float64).tiny,
        )

        scaled_bessel = ive(
            nu,
            kappa,
        )

        # log I_v(kappa)
        log_bessel = (
            np.log(scaled_bessel)
            + kappa
        )

        return (
            nu * np.log(kappa)
            - (n_features / 2.0)
            * np.log(2.0 * np.pi)
            - log_bessel
        )

    # ================================================================
    # Model parameter updates
    # ================================================================

    @staticmethod
    def _update_alpha(
        Z: np.ndarray,
    ) -> np.ndarray:
        """
        Equation (10b):

            alpha_h = (1 / n) sum_i z_ih
        """

        alpha = Z.mean(axis=0)

        # Numerical protection.
        alpha = np.maximum(
            alpha,
            np.finfo(np.float64).tiny,
        )

        alpha /= alpha.sum()

        return alpha

    def _update_W(
        self,
        X,
        Z: np.ndarray,
        mu: np.ndarray,
        kappa: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Equation (10a).

            v_jh = sum_i z_ih x_ij

            t_jh =
                kappa_h
                * mu_hh
                * v_jh

            w_jh = 1
                if h = argmax_l t_jl
        """

        # Shape: (n_features, n_clusters)
        V = X.T @ Z

        if sp.issparse(V):
            V = V.toarray()

        V = np.asarray(V)

        scores = V * (
            kappa * mu
        )[None, :]

        column_labels = np.argmax(
            scores,
            axis=1,
        )

        column_labels = self._repair_empty_clusters(
            column_labels,
            self.n_clusters,
            rng,
        )

        W = self._labels_to_membership(
            column_labels,
            self.n_clusters,
        )

        return W, column_labels

    @staticmethod
    def _update_mu(
        X,
        Z: np.ndarray,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        Equation (10c).

        For each co-cluster h:

            mu_hh
                =
                sign(s_h)
                /
                sqrt(sum_j w_jh)

        where:

            s_h =
                sum_i z_ih
                sum_j w_jh x_ij
        """

        XW = X @ W

        XW = np.asarray(
            XW,
            dtype=np.float64,
        )

        s = np.sum(
            Z * XW,
            axis=0,
        )

        column_sizes = W.sum(
            axis=0
        )

        if np.any(column_sizes <= 0):
            raise RuntimeError(
                "Encountered an empty column cluster."
            )

        signs = np.sign(s)

        # Degenerate directional statistic.
        # Choose the positive branch.
        signs[signs == 0] = 1.0

        return signs / np.sqrt(
            column_sizes
        )

    @staticmethod
    def _update_kappa(
        X,
        Z: np.ndarray,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        Equation (10d).

            kappa_h
                ≈
                (d * r_bar_h - r_bar_h^3)
                /
                (1 - r_bar_h^2)

        where:

            r_bar_h
                =
                ||r_h^w||
                /
                (
                    sum_i z_ih
                    *
                    sum_j w_jh
                )

        Due to the homogeneous block structure:

            ||r_h^w||
                =
                sqrt(sum_j w_jh)
                *
                abs(
                    sum_i z_ih
                    sum_j w_jh x_ij
                )
        """

        _, n_features = X.shape

        XW = X @ W

        XW = np.asarray(
            XW,
            dtype=np.float64,
        )

        s = np.sum(
            Z * XW,
            axis=0,
        )

        row_mass = Z.sum(
            axis=0
        )

        column_sizes = W.sum(
            axis=0
        )

        if np.any(row_mass <= 0):
            raise RuntimeError(
                "Encountered an empty row cluster."
            )

        if np.any(column_sizes <= 0):
            raise RuntimeError(
                "Encountered an empty column cluster."
            )

        r_norm = (
            np.sqrt(column_sizes)
            * np.abs(s)
        )

        r_bar = r_norm / (
            row_mass
            * column_sizes
        )

        r_bar = np.clip(
            r_bar,
            0.0,
            1.0 - 1e-10,
        )

        kappa = (
            n_features * r_bar
            - r_bar**3
        ) / (
            1.0
            - r_bar**2
        )

        return np.maximum(
            kappa,
            np.finfo(np.float64).tiny,
        )

    # ================================================================
    # E-step
    # ================================================================

    def _compute_log_joint(
        self,
        X,
        W: np.ndarray,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Compute:

            log p(x_i, z_i = h)

        for all samples i and clusters h.
        """

        _, n_features = X.shape

        # u_ih = sum_j x_ij w_jh
        U = X @ W

        U = np.asarray(
            U,
            dtype=np.float64,
        )

        log_normalizer = self._log_vmf_normalizer(
            kappa,
            n_features,
        )

        return (
            np.log(alpha)[None, :]
            + log_normalizer[None, :]
            + U
            * (kappa * mu)[None, :]
        )

    def _e_step(
        self,
        X,
        W: np.ndarray,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Soft posterior memberships.

            z_ih =
                p(z_i = h | x_i)
        """

        log_joint = self._compute_log_joint(
            X,
            W,
            mu,
            kappa,
            alpha,
        )

        log_prob = (
            log_joint
            - logsumexp(
                log_joint,
                axis=1,
                keepdims=True,
            )
        )

        Z = np.exp(log_prob)

        return Z, log_joint

    # ================================================================
    # Objective
    # ================================================================

    @staticmethod
    def _observed_log_likelihood(
        log_joint: np.ndarray,
    ) -> float:
        """
        Observed-data log-likelihood:

            sum_i log(
                sum_h
                alpha_h f_h(x_i)
            )
        """

        return float(
            np.sum(
                logsumexp(
                    log_joint,
                    axis=1,
                )
            )
        )

    # ================================================================
    # One initialization
    # ================================================================

    def _fit_single(
        self,
        X,
        rng: np.random.Generator,
        init_index: int,
    ) -> dict:

        n_samples, n_features = X.shape
        g = self.n_clusters

        # ------------------------------------------------------------
        # Initialize row partition Z
        # ------------------------------------------------------------

        row_labels = self._initialize_labels(
            n_samples,
            g,
            rng,
        )

        Z = self._labels_to_membership(
            row_labels,
            g,
        )

        # ------------------------------------------------------------
        # Initialize column partition W
        # ------------------------------------------------------------

        column_labels = self._initialize_labels(
            n_features,
            g,
            rng,
        )

        W = self._labels_to_membership(
            column_labels,
            g,
        )

        # ------------------------------------------------------------
        # Initial parameters
        # ------------------------------------------------------------

        alpha = self._update_alpha(Z)

        mu = self._update_mu(
            X,
            Z,
            W,
        )

        kappa = self._update_kappa(
            X,
            Z,
            W,
        )

        previous_Z = Z.copy()

        converged = False

        # ------------------------------------------------------------
        # EM / CEM loop
        # ------------------------------------------------------------

        for iteration in range(1, self.max_iter + 1):

            # ========================================================
            # E-step
            # ========================================================

            Z_soft, _ = self._e_step(
                X,
                W,
                mu,
                kappa,
                alpha,
            )

            # ========================================================
            # C-step for CEMb
            # ========================================================

            if self.algorithm == "CEMb":

                row_labels = np.argmax(
                    Z_soft,
                    axis=1,
                )

                row_labels = self._repair_empty_clusters(
                    row_labels,
                    g,
                    rng,
                )

                Z = self._labels_to_membership(
                    row_labels,
                    g,
                )

            else:
                Z = Z_soft

            # ========================================================
            # M-step: W
            # ========================================================

            W, column_labels = self._update_W(
                X,
                Z,
                mu,
                kappa,
                rng,
            )

            # ========================================================
            # M-step: alpha
            # ========================================================

            alpha = self._update_alpha(Z)

            # ========================================================
            # M-step: mu
            # ========================================================

            mu = self._update_mu(
                X,
                Z,
                W,
            )

            # ========================================================
            # M-step: kappa
            # ========================================================

            kappa = self._update_kappa(
                X,
                Z,
                W,
            )

            # ========================================================
            # Convergence
            # ========================================================

            delta = np.max(
                np.abs(
                    Z - previous_Z
                )
            )

            if self.verbose >= 2:
                print(
                    f"Initialization "
                    f"{init_index + 1}/{self.n_init}, "
                    f"iteration {iteration}/{self.max_iter}, "
                    f"delta={delta:.6e}"
                )

            if delta < self.tol:
                converged = True
                break

            previous_Z = Z.copy()

        # ------------------------------------------------------------
        # Compute final objective
        # ------------------------------------------------------------

        _, final_log_joint = self._e_step(
            X,
            W,
            mu,
            kappa,
            alpha,
        )

        lower_bound = self._observed_log_likelihood(
            final_log_joint
        )

        labels = np.argmax(
            Z,
            axis=1,
        )

        components = (
            W * mu[None, :]
        ).T

        return {
            "labels": labels,
            "column_labels": column_labels,
            "Z": Z,
            "W": W,
            "alpha": alpha,
            "mu": mu,
            "kappa": kappa,
            "components": components,
            "n_iter": iteration,
            "lower_bound": lower_bound,
            "converged": converged,
        }

    # ================================================================
    # sklearn API
    # ================================================================

    def fit(self, X, y=None, **kwargs):
        """
        Fit dbmovMFs to X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape
            (n_samples, n_features)

        y : ignored
            Present for sklearn compatibility.

        Returns
        -------
        self
        """

        self._validate_params()

        X = self._check_X(X)

        n_samples, n_features = X.shape

        if self.n_clusters > n_samples:
            raise ValueError(
                "n_clusters cannot exceed n_samples."
            )

        if self.n_clusters > n_features:
            raise ValueError(
                "n_clusters cannot exceed n_features."
            )

        X = self._normalize_rows(X)

        self.n_features_in_ = n_features

        # ============================================================
        # Random number generator
        # ============================================================

        if isinstance(
            self.random_state,
            np.random.Generator,
        ):
            master_rng = self.random_state

        else:
            master_rng = np.random.default_rng(
                self.random_state
            )

        best_result = None

        # ============================================================
        # Multiple initializations
        # ============================================================

        for init_index in range(self.n_init):

            # Independent RNG per initialization.
            seed = master_rng.integers(
                np.iinfo(np.int64).max
            )

            rng = np.random.default_rng(
                seed
            )

            result = self._fit_single(
                X,
                rng,
                init_index,
            )

            if self.verbose >= 1:
                print(
                    f"Initialization "
                    f"{init_index + 1}/{self.n_init}: "
                    f"log-likelihood="
                    f"{result['lower_bound']:.6f}, "
                    f"iterations="
                    f"{result['n_iter']}"
                )

            if (
                best_result is None
                or result["lower_bound"]
                > best_result["lower_bound"]
            ):
                best_result = result

        # ============================================================
        # Store fitted attributes
        # ============================================================

        self.labels_ = best_result["labels"]
        self.column_labels_ = best_result[
            "column_labels"
        ]

        self.membership_ = best_result["Z"]
        self.column_membership_ = best_result["W"]

        self.alpha_ = best_result["alpha"]
        self.mu_ = best_result["mu"]
        self.kappa_ = best_result["kappa"]

        self.components_ = best_result[
            "components"
        ]

        self.n_iter_ = best_result["n_iter"]
        self.lower_bound_ = best_result[
            "lower_bound"
        ]

        self.converged_ = best_result[
            "converged"
        ]

        return self

    def fit_predict(self, X, y=None, **kwargs):
        """
        Fit the model and return row-cluster labels.
        :param **kwargs:
        """

        return self.fit(
            X,
            y,
            **kwargs
        ).labels_

    def predict_proba(self, X) -> np.ndarray:
        """
        Return posterior row-cluster probabilities for new samples.
        """

        check_is_fitted(
            self,
            [
                "column_membership_",
                "mu_",
                "kappa_",
                "alpha_",
            ],
        )

        X = self._check_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but DBMOVMF was fitted with "
                f"{self.n_features_in_} features."
            )

        X = self._normalize_rows(X)

        Z, _ = self._e_step(
            X,
            self.column_membership_,
            self.mu_,
            self.kappa_,
            self.alpha_,
        )

        return Z

    def predict(self, X) -> np.ndarray:
        """
        Predict row-cluster labels for new samples.
        """

        probabilities = self.predict_proba(X)

        return np.argmax(
            probabilities,
            axis=1,
        )

    def score(self, X, y=None) -> float:
        """
        Return the mean observed-data log-likelihood per sample.

        Higher is better.
        """

        check_is_fitted(
            self,
            [
                "column_membership_",
                "mu_",
                "kappa_",
                "alpha_",
            ],
        )

        X = self._check_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but DBMOVMF was fitted with "
                f"{self.n_features_in_} features."
            )

        X = self._normalize_rows(X)

        log_joint = self._compute_log_joint(
            X,
            self.column_membership_,
            self.mu_,
            self.kappa_,
            self.alpha_,
        )

        return (
            self._observed_log_likelihood(
                log_joint
            )
            / X.shape[0]
        )