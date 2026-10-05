"""Group response vectors into themes with KMeans or HDBSCAN. Pure math: no database, no LLM.

A row's cluster is a number from 0 up (0 is the largest), or OUTLIER (-1) for a response that fits no
theme. Distances are cosine distances, like everywhere else in the project.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

OUTLIER = -1
K_RANGE = range(4, 9)  # the plan asks for 4 to 8 themes
OUTLIER_SIGMAS = 2.0  # KMeans: a response this many standard deviations farther than usual is an outlier
MIN_THEME_SIZE = 3  # KMeans: a cluster smaller than this is not a theme, so its members count as outliers
SEED = 0


@dataclass
class Clustering:
    labels: list[int]  # one per row: its cluster number, or OUTLIER
    method: str
    params: dict = field(default_factory=dict)
    silhouette: float | None = None  # how separated the clusters are, -1 to 1 (None with fewer than 2 clusters)

    @property
    def n_clusters(self) -> int:
        return len({label for label in self.labels if label != OUTLIER})

    @property
    def n_outliers(self) -> int:
        return sum(label == OUTLIER for label in self.labels)


def cosine_distance(vectors: np.ndarray, center: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(center)
    return 1 - (vectors @ center) / np.where(norms == 0, 1, norms)


def _renumber(labels: np.ndarray) -> list[int]:
    """Number the clusters by size, largest first. Outliers stay OUTLIER."""
    sizes = {int(c): int((labels == c).sum()) for c in set(labels.tolist()) if c != OUTLIER}
    new = {old: rank for rank, old in enumerate(sorted(sizes, key=lambda c: (-sizes[c], c)))}
    return [OUTLIER if label == OUTLIER else new[int(label)] for label in labels]


def _silhouette(X: np.ndarray, labels: list[int]) -> float | None:
    """Silhouette over the points that belong to a cluster. It needs at least 2 clusters."""
    keep = [i for i, label in enumerate(labels) if label != OUTLIER]
    kept = [labels[i] for i in keep]
    if len(set(kept)) < 2 or len(keep) <= len(set(kept)):
        return None
    return float(silhouette_score(X[keep], kept, metric="cosine"))


def centers(X: np.ndarray, labels: list[int]) -> dict[int, np.ndarray]:
    """The mean vector of each cluster."""
    array = np.array(labels)
    return {c: X[array == c].mean(axis=0) for c in sorted(set(labels)) if c != OUTLIER}


def distances_to_centers(X: np.ndarray, labels: list[int]) -> np.ndarray:
    """Each row's cosine distance to its own cluster's center. An outlier's is to the nearest center."""
    cs = centers(X, labels)
    result = np.zeros(len(X))
    for i, (row, label) in enumerate(zip(X, labels)):
        options = [cs[label]] if label != OUTLIER else list(cs.values())
        result[i] = min(float(cosine_distance(row[None, :], c)[0]) for c in options) if options else 0.0
    return result


def choose_k(X: np.ndarray, k_range=K_RANGE) -> tuple[int, dict[int, float]]:
    """Fit KMeans for each k and keep the one whose clusters are best separated (highest silhouette)."""
    scores = {}
    for k in k_range:
        if k >= len(X):
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels, metric="cosine"))
    best = max(scores, key=lambda k: scores[k])
    return best, scores


def cluster_kmeans(X: np.ndarray, k: int | None = None, min_size: int = MIN_THEME_SIZE) -> Clustering:
    """KMeans, then mark as outliers the responses unusually far from their own center and those in tiny clusters."""
    params: dict = {}
    if k is None:
        k, scores = choose_k(X)
        params["silhouette_by_k"] = {str(key): round(value, 4) for key, value in scores.items()}
    labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
    plain = [int(label) for label in labels]
    distance = distances_to_centers(X, plain)
    threshold = float(distance.mean() + OUTLIER_SIGMAS * distance.std())
    marked = np.where(distance > threshold, OUTLIER, labels)
    for cluster in set(marked.tolist()) - {OUTLIER}:
        if (marked == cluster).sum() < min_size:
            marked[marked == cluster] = OUTLIER
    params.update(k=k, outlier_threshold=round(threshold, 4), min_size=min_size)
    result = _renumber(marked)
    return Clustering(result, "kmeans", params, _silhouette(X, result))


def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 5, components: int = 20) -> Clustering:
    """HDBSCAN finds dense groups of any shape and calls the rest noise (the outliers).

    Density is hard to measure in 1536 dimensions, so the vectors are first reduced with PCA.
    """
    n_components = min(components, X.shape[1], len(X) - 1)
    reduced = PCA(n_components=n_components, random_state=SEED).fit_transform(X)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(reduced)
    result = _renumber(labels)
    params = {"min_cluster_size": min_cluster_size, "pca_components": n_components}
    return Clustering(result, "hdbscan", params, _silhouette(X, result))


def representatives(distances: np.ndarray, labels: list[int], cluster: int, n: int) -> list[int]:
    """Row numbers of the n members closest to the cluster's center, closest first."""
    members = [i for i, label in enumerate(labels) if label == cluster]
    return sorted(members, key=lambda i: distances[i])[:n]
