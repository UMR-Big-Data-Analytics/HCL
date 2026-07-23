from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

from vmf_hac import VmfHAC
from vmf_hac.baselines import SphericalKMeans, VonMisesFisherMixture

METHODS = [
    ("VmfHAC", lambda n_clusters: VmfHAC(n_clusters=n_clusters, gamma=0.05)),
    (
        "Ward",
        lambda n_clusters: AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        ),
    ),
    (
        "AverageCosine",
        lambda n_clusters: AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        ),
    ),
    (
        "CompleteCosine",
        lambda n_clusters: AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="complete",
        ),
    ),
    (
        "SingleCosine",
        lambda n_clusters: AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="single",
        ),
    ),
    (
        "KMeans",
        lambda n_clusters: KMeans(
            n_clusters=n_clusters,
        ),
    ),
    ("vMF-Mixture", lambda n_clusters: VonMisesFisherMixture(n_clusters=n_clusters)),
    (
        "Spherical-KMeans",
        lambda n_clusters: SphericalKMeans(
            n_clusters=n_clusters,
        ),
    ),
    (
        "Spectral",
        lambda n_clusters: SpectralClustering(
            n_clusters=n_clusters,
        ),
    ),
]
