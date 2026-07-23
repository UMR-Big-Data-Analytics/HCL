from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

from sklearn.base import BaseEstimator, ClusterMixin


@runtime_checkable
class Clusterer(Protocol):
    def fit(self, X: Any, y: Any = ...) -> Self: ...

    def fit_predict(self, X: Any, y: Any = ...) -> Any: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> Self: ...


_cluster_mixin: type[ClusterMixin] = ClusterMixin
_base_estimator: type[BaseEstimator] = BaseEstimator
