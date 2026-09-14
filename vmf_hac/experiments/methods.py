from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.pipeline import Pipeline
from umap import UMAP

from vmf_hac import VmfHAC
from vmf_hac.baselines import SphericalKMeans, VonMisesFisherMixture

METHODS = [
    ("VmfHAC", lambda n_clusters, **kwargs: VmfHAC(n_clusters=n_clusters, gamma=0.075, **kwargs)),
    (
        "Ward",
        lambda n_clusters, **kwargs: AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        ),
    ),
    (
        "AverageCosine",
        lambda n_clusters, **kwargs: AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
            **kwargs,
        ),
    ),
    (
        "CompleteCosine",
        lambda n_clusters, **kwargs: AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="complete",
            **kwargs,
        ),
    ),
    (
        "SingleCosine",
        lambda n_clusters, **kwargs: AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="single",
            **kwargs,
        ),
    ),
    (
        "KMeans",
        lambda n_clusters, **kwargs: KMeans(
            n_clusters=n_clusters,
            **kwargs,
        ),
    ),
    ("vMF-Mixture", lambda n_clusters, **kwargs: VonMisesFisherMixture(n_clusters=n_clusters, **kwargs)),
    (
        "Spherical-KMeans",
        lambda n_clusters, **kwargs: SphericalKMeans(
            n_clusters=n_clusters,
            **kwargs,
        ),
    ),
    (
        "Spectral",
        lambda n_clusters, **kwargs: SpectralClustering(
            n_clusters=n_clusters,
            **kwargs,
        ),
    ),
    (
        "HDBSCAN",
        lambda n_clusters, **kwargs: Pipeline(
            [
                ("umap", UMAP(n_components=15, metric="cosine", random_state=42, n_jobs=1)),
                ("hdbscan", HDBSCAN(metric="euclidean", copy=True, **kwargs)),
            ]
        ),
    ),
]
