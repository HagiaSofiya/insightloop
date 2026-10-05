import numpy as np
import pytest

from app.clustering import (
    OUTLIER, Clustering, centers, choose_k, cluster_hdbscan, cluster_kmeans, cosine_distance,
    distances_to_centers, representatives,
)

DIMS = 20


def blobs(sizes, noise=0.05, seed=0, extra=None):
    """Tight groups of points around separate axes. Returns the matrix and the true group of each row."""
    rng = np.random.RandomState(seed)
    rows, truth = [], []
    for group, size in enumerate(sizes):
        center = np.zeros(DIMS)
        center[group] = 1.0
        for _ in range(size):
            rows.append(center + rng.normal(0, noise, DIMS))
            truth.append(group)
    for point in extra or []:
        rows.append(point)
        truth.append(OUTLIER)
    return np.array(rows), truth


def groups_of(labels, truth):
    """Which true groups ended up in each cluster, as sets."""
    return {frozenset(t for l2, t in zip(labels, truth) if l2 == c) for c in set(labels) - {OUTLIER}}


def test_kmeans_recovers_separate_groups_exactly():
    X, truth = blobs([12, 12, 12])

    result = cluster_kmeans(X, k=3)

    assert result.n_clusters == 3 and result.n_outliers == 0
    assert groups_of(result.labels, truth) == {frozenset({0}), frozenset({1}), frozenset({2})}  # each cluster is one group
    assert result.silhouette > 0.8  # well separated
    assert result.method == "kmeans" and result.params["k"] == 3


def test_clusters_are_numbered_by_size_largest_first():
    X, truth = blobs([6, 20, 10])

    result = cluster_kmeans(X, k=3)

    sizes = [result.labels.count(c) for c in range(3)]
    assert sizes == [20, 10, 6]


def test_the_best_k_is_the_one_with_the_best_separation():
    X, _ = blobs([12, 12, 12])

    best, scores = choose_k(X, range(2, 7))

    assert best == 3
    assert scores[3] > max(scores[k] for k in (2, 4, 5, 6))


def test_without_a_k_kmeans_chooses_one_and_records_the_scores():
    X, _ = blobs([10, 10, 10, 10, 10])

    result = cluster_kmeans(X)  # the default range is 4 to 8

    assert 4 <= result.params["k"] <= 8
    assert set(result.params["silhouette_by_k"]) <= {"4", "5", "6", "7", "8"}


def test_a_response_far_from_every_group_is_an_outlier():
    far = np.zeros(DIMS)
    far[15] = 1.0  # points along an axis none of the groups use
    X, truth = blobs([12, 12, 12], extra=[far])

    result = cluster_kmeans(X, k=3)

    assert result.labels[-1] == OUTLIER
    assert result.n_outliers >= 1
    assert OUTLIER not in result.labels[:36]  # the normal points all stay in their groups


def test_a_cluster_below_the_minimum_size_is_not_a_theme():
    pair = [np.eye(DIMS)[10], np.eye(DIMS)[10] + 0.01]  # two close points, far from everything else
    X, truth = blobs([12, 12, 12], extra=pair)

    result = cluster_kmeans(X, k=4)

    assert result.labels[-2:] == [OUTLIER, OUTLIER]
    assert result.n_clusters == 3
    assert result.params["min_size"] == 3


def test_hdbscan_finds_dense_groups_and_calls_scattered_points_noise():
    rng = np.random.RandomState(1)
    scattered = [rng.normal(0, 1, DIMS) for _ in range(6)]
    X, truth = blobs([15, 15, 15], noise=0.03, extra=scattered)

    result = cluster_hdbscan(X, min_cluster_size=5)

    assert result.method == "hdbscan" and result.n_clusters == 3
    assert groups_of(result.labels, truth) == {frozenset({0}), frozenset({1}), frozenset({2})}
    assert result.n_outliers >= 3  # most of the scattered points are noise
    assert result.params["min_cluster_size"] == 5


def test_hdbscan_may_find_nothing_in_data_with_no_dense_groups():
    X = np.random.RandomState(2).normal(0, 1, (30, DIMS))

    result = cluster_hdbscan(X, min_cluster_size=4)

    assert result.n_clusters == 0 and result.n_outliers == 30 and result.silhouette is None


def test_silhouette_needs_two_clusters():
    X, _ = blobs([20])

    assert cluster_kmeans(X, k=1).silhouette is None  # one cluster, nothing to separate from


def test_cosine_distance_ignores_length():
    vectors = np.array([[1.0, 0.0], [10.0, 0.0], [0.0, 3.0]])

    assert cosine_distance(vectors, np.array([2.0, 0.0])).tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_distances_to_centers_and_representatives_pick_the_most_typical_answers():
    X = np.array([[1.0, 0.0], [1.0, 0.05], [1.0, 0.3], [0.0, 1.0], [0.05, 1.0]])
    labels = [0, 0, 0, 1, 1]

    distance = distances_to_centers(X, labels)

    assert representatives(distance, labels, 0, 2) == [1, 0]  # the two nearest to cluster 0's center, nearest first
    assert representatives(distance, labels, 1, 5) == [3, 4] or representatives(distance, labels, 1, 5) == [4, 3]
    assert set(centers(X, labels)) == {0, 1}


def test_an_outliers_distance_is_to_the_nearest_center():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    labels = [0, 0, 1, 1, OUTLIER]

    distance = distances_to_centers(X, labels)

    assert distance[4] == pytest.approx(1 - 1 / np.sqrt(2))  # 45 degrees from both centers


def test_the_clustering_summary_counts_clusters_and_outliers():
    result = Clustering([0, 0, 1, OUTLIER, OUTLIER], "kmeans")

    assert (result.n_clusters, result.n_outliers) == (2, 2)
