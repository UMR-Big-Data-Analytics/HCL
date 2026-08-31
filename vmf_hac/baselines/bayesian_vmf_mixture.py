from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.special import digamma, ive, logsumexp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array, check_is_fitted


class BayesianVMFMixture(BaseEstimator, ClusterMixin):
    """
    Bayesian von Mises-Fisher mixture model (B-vMFmix).

    Implementation of the finite Bayesian vMF mixture from:

        Gopal, S. and Yang, Y. (2014).
        Von Mises-Fisher Clustering Models.
        Proceedings of ICML 2014, pp. 154-162.

    The model is:

        pi ~ Dirichlet(alpha)

        mu_k ~ vMF(mu0, C0)

        kappa_k ~ LogNormal(m, sigma^2)

        z_i ~ Categorical(pi)

        x_i ~ vMF(mu_{z_i}, kappa_{z_i})

    Variational distributions:

        q(pi)   = Dirichlet(rho)
        q(mu_k) = vMF(psi_k, gamma_k)
        q(z_i)  = Categorical(lambda_i)

    The posterior over kappa_k is approximated using the
    partial-MCMC scheme described in Section 3.1.1 of the
    paper.

    Parameters
    ----------
    n_clusters : int
        Number of mixture components.

    alpha : float or array-like, default=1.0
        Dirichlet prior concentration. A scalar specifies a
        symmetric Dirichlet prior.

    mu0 : array-like of shape (n_features,), optional
        Common prior mean direction.

        If None, the normalized global data mean is used.

    C0 : float, default=1.0
        Concentration of the common vMF prior over component
        mean directions.

    m : float, default=0.0
        Mean of log(kappa) under the log-normal prior.

    sigma : float, default=1.0
        Standard deviation of log(kappa) under the log-normal
        prior.

    max_iter : int, default=100
        Maximum number of variational iterations.

    tol : float, default=1e-5
        Convergence tolerance based on maximum change in
        responsibilities.

    n_init : int, default=5
        Number of independent initializations.

    init : {"kmeans", "random"}, default="kmeans"
        Initialization method for component directions.

    kappa_init : float, default=10.0
        Initial concentration.

    n_mcmc : int, default=20
        Number of MCMC steps used for each kappa update.

    mcmc_burn_in : int, default=10
        Number of initial MCMC samples discarded.

    proposal_std : float, default=0.25
        Standard deviation of the random-walk proposal in
        log(kappa)-space.

    min_kappa : float, default=1e-6
        Minimum permitted concentration.

    max_kappa : float, default=1e6
        Maximum permitted concentration.

    random_state : int, optional
        Random seed.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Hard cluster assignments.

    responsibilities_ : ndarray of shape
        (n_samples, n_clusters)
        Variational posterior q(z_i).

    rho_ : ndarray of shape (n_clusters,)
        Dirichlet posterior parameters.

    psi_ : ndarray of shape
        (n_clusters, n_features)
        Posterior mean directions.

    gamma_ : ndarray of shape (n_clusters,)
        Posterior vMF concentrations for q(mu_k).

    kappa_ : ndarray of shape (n_clusters,)
        Posterior estimates E_q[kappa_k], approximated using
        MCMC samples.

    kappa_samples_ : list of ndarray
        Final posterior samples for each kappa_k.

    n_iter_ : int
        Number of variational iterations.

    converged_ : bool
        Whether convergence criterion was satisfied.

    Notes
    -----
    Input samples are L2-normalized internally.

    This estimator implements the finite B-vMFmix model from
    Section 3.1 and uses the partial-MCMC inference scheme for
    concentration parameters described in Section 3.1.1.

    The sampling approximation means that the monitored
    objective is not guaranteed to be a strict variational lower
    bound, as explicitly noted by Gopal and Yang.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        alpha: float | np.ndarray = 1.0,
        mu0: Optional[np.ndarray] = None,
        C0: float = 1.0,
        m: float = 0.0,
        sigma: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-5,
        n_init: int = 5,
        init: str = "kmeans",
        kappa_init: float = 10.0,
        n_mcmc: int = 20,
        mcmc_burn_in: int = 10,
        proposal_std: float = 0.25,
        min_kappa: float = 1e-6,
        max_kappa: float = 1e6,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mu0 = mu0
        self.C0 = C0
        self.m = m
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.kappa_init = kappa_init
        self.n_mcmc = n_mcmc
        self.mcmc_burn_in = mcmc_burn_in
        self.proposal_std = proposal_std
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_params(self):
        if self.n_clusters < 1:
            raise ValueError(
                "n_clusters must be >= 1."
            )

        if self.C0 < 0:
            raise ValueError(
                "C0 must be >= 0."
            )

        if self.sigma <= 0:
            raise ValueError(
                "sigma must be > 0."
            )

        if self.max_iter < 1:
            raise ValueError(
                "max_iter must be >= 1."
            )

        if self.n_init < 1:
            raise ValueError(
                "n_init must be >= 1."
            )

        if self.n_mcmc < 1:
            raise ValueError(
                "n_mcmc must be >= 1."
            )

        if self.mcmc_burn_in < 0:
            raise ValueError(
                "mcmc_burn_in must be >= 0."
            )

        if self.proposal_std <= 0:
            raise ValueError(
                "proposal_std must be > 0."
            )

        if self.min_kappa <= 0:
            raise ValueError(
                "min_kappa must be > 0."
            )

        if self.max_kappa <= self.min_kappa:
            raise ValueError(
                "max_kappa must exceed min_kappa."
            )

        if self.init not in ("kmeans", "random"):
            raise ValueError(
                "init must be 'kmeans' or 'random'."
            )

    @staticmethod
    def _normalize_rows(X):
        norms = np.linalg.norm(
            X,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "X contains zero-norm samples."
            )

        return X / norms

    # ------------------------------------------------------------------
    # vMF functions
    # ------------------------------------------------------------------

    @staticmethod
    def _log_vmf_normalizer(
        kappa: np.ndarray,
        dimension: int,
    ) -> np.ndarray:
        """
        log C_D(kappa).

        Uses exponentially scaled Bessel functions:

            ive(v, x) = exp(-|x|) I_v(x)

        to avoid overflow.
        """

        kappa = np.asarray(
            kappa,
            dtype=np.float64,
        )

        kappa = np.maximum(
            kappa,
            np.finfo(np.float64).tiny,
        )

        nu = dimension / 2.0 - 1.0

        scaled_bessel = ive(
            nu,
            kappa,
        )

        scaled_bessel = np.maximum(
            scaled_bessel,
            np.finfo(np.float64).tiny,
        )

        log_bessel = (
            np.log(scaled_bessel)
            + kappa
        )

        return (
            nu * np.log(kappa)
            - dimension / 2.0 * np.log(
                2.0 * np.pi
            )
            - log_bessel
        )

    @staticmethod
    def _vmf_mean_resultant_length(
        dimension: int,
        concentration: np.ndarray,
    ) -> np.ndarray:
        """
        A_D(kappa) = I_{D/2}(kappa) / I_{D/2 - 1}(kappa).

        This gives:

            E[mu]
            =
            A_D(gamma) * psi

        for:

            mu ~ vMF(psi, gamma).
        """

        concentration = np.asarray(
            concentration,
            dtype=np.float64,
        )

        nu = dimension / 2.0 - 1.0

        numerator = ive(
            nu + 1.0,
            concentration,
        )

        denominator = ive(
            nu,
            concentration,
        )

        denominator = np.maximum(
            denominator,
            np.finfo(np.float64).tiny,
        )

        return numerator / denominator

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def _make_alpha(self):
        if np.isscalar(self.alpha):
            if self.alpha <= 0:
                raise ValueError(
                    "alpha must be > 0."
                )

            return np.full(
                self.n_clusters,
                float(self.alpha),
            )

        alpha = np.asarray(
            self.alpha,
            dtype=np.float64,
        )

        if alpha.shape != (
            self.n_clusters,
        ):
            raise ValueError(
                "alpha must have shape "
                "(n_clusters,)."
            )

        if np.any(alpha <= 0):
            raise ValueError(
                "All alpha values must be > 0."
            )

        return alpha

    def _make_mu0(
        self,
        X: np.ndarray,
    ):
        if self.mu0 is None:

            mu0 = X.sum(
                axis=0,
            )

            norm = np.linalg.norm(mu0)

            if norm == 0:
                mu0 = np.zeros(
                    X.shape[1],
                )

                mu0[0] = 1.0

            else:
                mu0 /= norm

            return mu0

        mu0 = np.asarray(
            self.mu0,
            dtype=np.float64,
        )

        if mu0.shape != (
            X.shape[1],
        ):
            raise ValueError(
                "mu0 has incompatible dimension."
            )

        norm = np.linalg.norm(mu0)

        if norm == 0:
            raise ValueError(
                "mu0 must have non-zero norm."
            )

        return mu0 / norm

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ):
        n_samples, dimension = X.shape
        K = self.n_clusters

        if self.init == "kmeans":

            seed = int(
                rng.integers(
                    np.iinfo(np.int32).max,
                )
            )

            km = KMeans(
                n_clusters=K,
                n_init=1,
                random_state=seed,
            )

            labels = km.fit_predict(X)

            psi = np.zeros(
                (K, dimension),
            )

            for k in range(K):

                members = X[
                    labels == k
                ]

                if len(members) == 0:

                    vector = rng.normal(
                        size=dimension,
                    )

                else:

                    vector = members.sum(
                        axis=0,
                    )

                norm = np.linalg.norm(
                    vector
                )

                if norm == 0:

                    vector = rng.normal(
                        size=dimension,
                    )

                    norm = np.linalg.norm(
                        vector
                    )

                psi[k] = vector / norm

            scores = X @ psi.T

            log_lambda = scores

            log_lambda -= logsumexp(
                log_lambda,
                axis=1,
                keepdims=True,
            )

            responsibilities = np.exp(
                log_lambda
            )

        else:

            psi = rng.normal(
                size=(K, dimension),
            )

            psi /= np.linalg.norm(
                psi,
                axis=1,
                keepdims=True,
            )

            scores = X @ psi.T

            scores += 0.01 * rng.normal(
                size=scores.shape,
            )

            scores -= logsumexp(
                scores,
                axis=1,
                keepdims=True,
            )

            responsibilities = np.exp(
                scores
            )

        kappa = np.full(
            K,
            self.kappa_init,
            dtype=np.float64,
        )

        return (
            responsibilities,
            psi,
            kappa,
        )

    # ------------------------------------------------------------------
    # Variational updates
    # ------------------------------------------------------------------

    @staticmethod
    def _update_rho(
        alpha: np.ndarray,
        responsibilities: np.ndarray,
    ):
        """
        rho_k = alpha_k + sum_i lambda_ik
        """

        return (
            alpha
            + responsibilities.sum(axis=0)
        )

    def _update_mu_posterior(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray,
        expected_kappa: np.ndarray,
        mu0: np.ndarray,
    ):
        """
        Paper:

            R_k =
                E[kappa_k]
                sum_i E[z_ik] x_i
                + C0 mu0

            psi_k = R_k / ||R_k||

            gamma_k = ||R_k||
        """

        sufficient_statistics = (
            responsibilities.T @ X
        )

        R = (
            expected_kappa[:, None]
            * sufficient_statistics
            + self.C0
            * mu0[None, :]
        )

        gamma = np.linalg.norm(
            R,
            axis=1,
        )

        psi = np.empty_like(R)

        for k in range(
            self.n_clusters
        ):

            if gamma[k] <= 1e-15:

                psi[k] = mu0
                gamma[k] = 0.0

            else:

                psi[k] = (
                    R[k]
                    / gamma[k]
                )

        return psi, gamma

    def _update_responsibilities(
        self,
        X: np.ndarray,
        rho: np.ndarray,
        psi: np.ndarray,
        gamma: np.ndarray,
        expected_kappa: np.ndarray,
        expected_log_C: np.ndarray,
    ):
        """
        lambda_ik proportional to

            exp(
                E[log pi_k]
                +
                E[log C_D(kappa_k)]
                +
                E[kappa_k]
                x_i^T E[mu_k]
            )

        where:

            E[mu_k]
                =
                A_D(gamma_k) psi_k
        """

        dimension = X.shape[1]

        expected_log_pi = (
            digamma(rho)
            - digamma(
                np.sum(rho)
            )
        )

        A_gamma = (
            self._vmf_mean_resultant_length(
                dimension,
                gamma,
            )
        )

        expected_mu = (
            A_gamma[:, None]
            * psi
        )

        log_responsibilities = (
            expected_log_pi[None, :]
            + expected_log_C[None, :]
            + (
                X @ expected_mu.T
            )
            * expected_kappa[None, :]
        )

        log_responsibilities -= logsumexp(
            log_responsibilities,
            axis=1,
            keepdims=True,
        )

        return np.exp(
            log_responsibilities
        )

    # ------------------------------------------------------------------
    # Kappa posterior
    # ------------------------------------------------------------------

    def _log_kappa_conditional(
        self,
        log_kappa: float,
        *,
        dimension: int,
        responsibility_sum: float,
        directional_statistic: float,
    ) -> float:
        """
        Log of the proportional conditional distribution from
        Equation (1) of Gopal & Yang.

        p(kappa_k | ...)
        proportional to

            exp(
                N_k log C_D(kappa_k)
                +
                kappa_k S_k
            )
            *
            LogNormal(kappa_k | m, sigma^2)

        Sampling is performed in eta = log(kappa) coordinates.

        The target density therefore includes the Jacobian:

            p(eta) = p(kappa) * kappa.
        """

        if not np.isfinite(
            log_kappa
        ):
            return -np.inf

        kappa = np.exp(
            log_kappa
        )

        if (
            kappa < self.min_kappa
            or kappa > self.max_kappa
        ):
            return -np.inf

        log_C = (
            self._log_vmf_normalizer(
                np.array([kappa]),
                dimension,
            )[0]
        )

        # Log-normal prior for kappa.
        #
        # log p(kappa)
        #
        # =
        # -log(kappa)
        # -(log(kappa)-m)^2/(2 sigma^2)
        # -log(sigma sqrt(2 pi))
        log_prior_kappa = (
            -log_kappa
            - (
                (log_kappa - self.m) ** 2
                / (
                    2.0
                    * self.sigma ** 2
                )
            )
            - np.log(
                self.sigma
                * np.sqrt(
                    2.0 * np.pi
                )
            )
        )

        # Jacobian for eta = log(kappa).
        log_jacobian = log_kappa

        return (
            responsibility_sum
            * log_C
            + kappa
            * directional_statistic
            + log_prior_kappa
            + log_jacobian
        )

    def _sample_kappa(
        self,
        current_kappa: float,
        *,
        dimension: int,
        responsibility_sum: float,
        directional_statistic: float,
        rng: np.random.Generator,
    ):
        """
        Random-walk Metropolis in log(kappa)-space.

        The paper states that a log-normal proposal around
        the current iterate was used.
        """

        eta = np.log(
            np.clip(
                current_kappa,
                self.min_kappa,
                self.max_kappa,
            )
        )

        current_log_target = (
            self._log_kappa_conditional(
                eta,
                dimension=dimension,
                responsibility_sum=responsibility_sum,
                directional_statistic=directional_statistic,
            )
        )

        samples = []

        total_steps = (
            self.mcmc_burn_in
            + self.n_mcmc
        )

        accepted = 0

        for step in range(
            total_steps
        ):

            proposal_eta = (
                eta
                + rng.normal(
                    scale=self.proposal_std
                )
            )

            proposal_log_target = (
                self._log_kappa_conditional(
                    proposal_eta,
                    dimension=dimension,
                    responsibility_sum=responsibility_sum,
                    directional_statistic=directional_statistic,
                )
            )

            log_acceptance = (
                proposal_log_target
                - current_log_target
            )

            if (
                np.log(rng.random())
                < log_acceptance
            ):

                eta = proposal_eta

                current_log_target = (
                    proposal_log_target
                )

                accepted += 1

            if step >= self.mcmc_burn_in:

                samples.append(
                    np.exp(eta)
                )

        samples = np.asarray(
            samples,
            dtype=np.float64,
        )

        return (
            samples,
            accepted / total_steps,
        )

    def _update_kappa_posteriors(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray,
        psi: np.ndarray,
        gamma: np.ndarray,
        current_kappa: np.ndarray,
        rng: np.random.Generator,
    ):
        """
        Estimate q(kappa_k) by the partial-MCMC procedure.

        For the conditional from Equation (1):

            S_k =
                sum_i E[z_ik]
                x_i^T E[mu_k]

        The expectation E[mu_k] is:

            A_D(gamma_k) psi_k.
        """

        dimension = X.shape[1]

        A_gamma = (
            self._vmf_mean_resultant_length(
                dimension,
                gamma,
            )
        )

        expected_mu = (
            A_gamma[:, None]
            * psi
        )

        expected_kappa = np.empty(
            self.n_clusters
        )

        expected_log_C = np.empty(
            self.n_clusters
        )

        posterior_samples = []

        acceptance_rates = []

        responsibility_sums = (
            responsibilities.sum(axis=0)
        )

        directional_statistics = np.sum(
            (
                responsibilities.T @ X
            )
            * expected_mu,
            axis=1,
        )

        for k in range(
            self.n_clusters
        ):

            samples, acceptance = (
                self._sample_kappa(
                    current_kappa[k],
                    dimension=dimension,
                    responsibility_sum=float(
                        responsibility_sums[k]
                    ),
                    directional_statistic=float(
                        directional_statistics[k]
                    ),
                    rng=rng,
                )
            )

            expected_kappa[k] = np.mean(
                samples
            )

            expected_log_C[k] = np.mean(
                self._log_vmf_normalizer(
                    samples,
                    dimension,
                )
            )

            posterior_samples.append(
                samples
            )

            acceptance_rates.append(
                acceptance
            )

        return (
            expected_kappa,
            expected_log_C,
            posterior_samples,
            np.asarray(
                acceptance_rates
            ),
        )

    # ------------------------------------------------------------------
    # Objective proxy
    # ------------------------------------------------------------------

    def _expected_complete_log_likelihood(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray,
        rho: np.ndarray,
        psi: np.ndarray,
        gamma: np.ndarray,
        expected_kappa: np.ndarray,
        expected_log_C: np.ndarray,
    ):
        """
        Diagnostic objective.

        Because kappa expectations are estimated using partial
        MCMC, this quantity is not claimed to be a strict ELBO.
        """

        dimension = X.shape[1]

        expected_log_pi = (
            digamma(rho)
            - digamma(
                np.sum(rho)
            )
        )

        A_gamma = (
            self._vmf_mean_resultant_length(
                dimension,
                gamma,
            )
        )

        expected_mu = (
            A_gamma[:, None]
            * psi
        )

        log_terms = (
            expected_log_pi[None, :]
            + expected_log_C[None, :]
            + (
                X @ expected_mu.T
            )
            * expected_kappa[None, :]
        )

        entropy = (
            -np.sum(
                responsibilities
                * np.log(
                    np.maximum(
                        responsibilities,
                        np.finfo(
                            np.float64
                        ).tiny,
                    )
                )
            )
        )

        return float(
            np.sum(
                responsibilities
                * log_terms
            )
            + entropy
        )

    # ------------------------------------------------------------------
    # Single initialization
    # ------------------------------------------------------------------

    def _fit_single(
        self,
        X: np.ndarray,
        alpha: np.ndarray,
        mu0: np.ndarray,
        rng: np.random.Generator,
        init_index: int,
    ):
        (
            responsibilities,
            psi,
            current_kappa,
        ) = self._initialize(
            X,
            rng,
        )

        previous_responsibilities = (
            responsibilities.copy()
        )

        rho = self._update_rho(
            alpha,
            responsibilities,
        )

        gamma = np.ones(
            self.n_clusters
        )

        expected_log_C = (
            self._log_vmf_normalizer(
                current_kappa,
                X.shape[1],
            )
        )

        posterior_samples = None

        acceptance_rates = None

        converged = False

        objective = -np.inf

        for iteration in range(
            1,
            self.max_iter + 1,
        ):

            # ------------------------------------------------------
            # Update q(pi)
            # ------------------------------------------------------

            rho = self._update_rho(
                alpha,
                responsibilities,
            )

            # ------------------------------------------------------
            # Update q(mu_k)
            # ------------------------------------------------------

            psi, gamma = (
                self._update_mu_posterior(
                    X,
                    responsibilities,
                    current_kappa,
                    mu0,
                )
            )

            # ------------------------------------------------------
            # Update q(kappa_k)
            # ------------------------------------------------------

            (
                expected_kappa,
                expected_log_C,
                posterior_samples,
                acceptance_rates,
            ) = self._update_kappa_posteriors(
                X,
                responsibilities,
                psi,
                gamma,
                current_kappa,
                rng,
            )

            current_kappa = (
                expected_kappa
            )

            # ------------------------------------------------------
            # Update q(z_i)
            # ------------------------------------------------------

            responsibilities = (
                self._update_responsibilities(
                    X,
                    rho,
                    psi,
                    gamma,
                    expected_kappa,
                    expected_log_C,
                )
            )

            # ------------------------------------------------------
            # Convergence
            # ------------------------------------------------------

            delta = np.max(
                np.abs(
                    responsibilities
                    - previous_responsibilities
                )
            )

            objective = (
                self._expected_complete_log_likelihood(
                    X,
                    responsibilities,
                    rho,
                    psi,
                    gamma,
                    expected_kappa,
                    expected_log_C,
                )
            )

            if self.verbose >= 2:

                print(
                    f"init={init_index + 1}/{self.n_init} "
                    f"iter={iteration} "
                    f"delta={delta:.3e} "
                    f"objective={objective:.6f} "
                    f"kappa_mean="
                    f"{expected_kappa.mean():.3f} "
                    f"acceptance="
                    f"{acceptance_rates.mean():.3f}"
                )

            if delta < self.tol:

                converged = True

                break

            previous_responsibilities = (
                responsibilities.copy()
            )

        return {
            "responsibilities": responsibilities,
            "rho": rho,
            "psi": psi,
            "gamma": gamma,
            "kappa": expected_kappa,
            "log_C": expected_log_C,
            "kappa_samples": posterior_samples,
            "acceptance_rates": acceptance_rates,
            "objective": objective,
            "n_iter": iteration,
            "converged": converged,
        }

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y=None, **kwargs):
        """
        Fit the Bayesian vMF mixture.
        """

        self._validate_params()

        X = check_array(
            X,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=True,
        )

        if (
            X.shape[0]
            < self.n_clusters
        ):
            raise ValueError(
                "n_clusters cannot exceed "
                "n_samples."
            )

        X = self._normalize_rows(
            X
        )

        n_samples, dimension = X.shape

        self.n_features_in_ = (
            dimension
        )

        alpha = self._make_alpha()

        mu0 = self._make_mu0(
            X
        )

        master_rng = (
            np.random.default_rng(
                self.random_state
            )
        )

        best_result = None

        for init_index in range(
            self.n_init
        ):

            rng = np.random.default_rng(
                master_rng.integers(
                    np.iinfo(
                        np.int64
                    ).max
                )
            )

            result = self._fit_single(
                X,
                alpha,
                mu0,
                rng,
                init_index,
            )

            if self.verbose >= 1:

                print(
                    f"Initialization "
                    f"{init_index + 1}/"
                    f"{self.n_init}: "
                    f"objective="
                    f"{result['objective']:.6f}, "
                    f"iterations="
                    f"{result['n_iter']}, "
                    f"converged="
                    f"{result['converged']}"
                )

            if (
                best_result is None
                or result["objective"]
                > best_result["objective"]
            ):

                best_result = result

        self.responsibilities_ = (
            best_result[
                "responsibilities"
            ]
        )

        self.labels_ = np.argmax(
            self.responsibilities_,
            axis=1,
        )

        self.rho_ = best_result[
            "rho"
        ]

        self.psi_ = best_result[
            "psi"
        ]

        self.gamma_ = best_result[
            "gamma"
        ]

        self.kappa_ = best_result[
            "kappa"
        ]

        self.log_C_ = best_result[
            "log_C"
        ]

        self.kappa_samples_ = (
            best_result[
                "kappa_samples"
            ]
        )

        self.acceptance_rates_ = (
            best_result[
                "acceptance_rates"
            ]
        )

        self.objective_ = best_result[
            "objective"
        ]

        self.n_iter_ = best_result[
            "n_iter"
        ]

        self.converged_ = best_result[
            "converged"
        ]

        self.alpha_ = alpha

        self.mu0_ = mu0

        return self

    def fit_predict(
        self,
        X,
        y=None,
            **kwargs
    ):
        return self.fit(
            X,
            y,
            **kwargs
        ).labels_

    def predict_proba(
        self,
        X,
    ):
        """
        Posterior cluster probabilities for new observations.

        Uses the learned variational posterior parameters.
        """

        check_is_fitted(
            self,
            [
                "rho_",
                "psi_",
                "gamma_",
                "kappa_",
                "log_C_",
            ],
        )

        X = check_array(
            X,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=True,
        )

        if (
            X.shape[1]
            != self.n_features_in_
        ):
            raise ValueError(
                "X has incompatible number "
                "of features."
            )

        X = self._normalize_rows(
            X
        )

        return self._update_responsibilities(
            X,
            self.rho_,
            self.psi_,
            self.gamma_,
            self.kappa_,
            self.log_C_,
        )

    def predict(
        self,
        X,
    ):
        return np.argmax(
            self.predict_proba(
                X
            ),
            axis=1,
        )