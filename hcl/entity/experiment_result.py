from typing import Any

from pydantic import BaseModel, Field


class ExperimentResult(BaseModel):
    clusterer_name: str
    dataset_name: str
    dim: int
    nmi: float
    ari: float
    ami: float
    v_measure: float
    v_measure_singleton: float
    v_measure_nearest: float
    ami_nearest: float
    ari_nearest: float
    noise_fraction: float
    homogeneity: float
    completeness: float
    fowlkes_mallows: float
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    n_clusters: int
    n_true_clusters: int
    running_time_s: float
    params: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
