import numpy as np

from embedding_space_exploration.battery.cluster import (
    build_labels_frame,
    cluster_metrics_by_split,
    cluster_sizes,
    fit_clusters,
)
from embedding_space_exploration.splits import TEST, TRAIN, VAL

N_CLUSTERS = 2
EXPECTED_SPLIT_METRIC_ROWS = 2


def _two_blobs(n_per=30, seed=0):
    """Two well-separated blobs; rows 0..n-1 = blob A, n..2n-1 = blob B."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[5, 0], scale=0.1, size=(n_per, 2))
    b = rng.normal(loc=[-5, 0], scale=0.1, size=(n_per, 2))
    return np.vstack([a, b])


def test_held_out_patients_are_assigned_via_predict():
    matrix = _two_blobs(n_per=30)
    # both blobs seen by both splits
    split_labels = np.array([TRAIN, VAL] * 30)
    model = fit_clusters(matrix[split_labels == TRAIN], N_CLUSTERS)

    labels = model.predict(matrix)

    assert len(labels) == len(matrix)  # every patient labelled, incl. held-out
    assert set(np.unique(labels)) == {0, 1}


def test_cluster_metrics_reported_per_split():
    matrix = _two_blobs(n_per=30)
    # test rows present but must be excluded from the reported metrics
    split_labels = np.array([TRAIN, VAL, TEST] * 20)
    labels = fit_clusters(matrix[split_labels == TRAIN], N_CLUSTERS).predict(matrix)

    metrics = cluster_metrics_by_split(matrix, labels, split_labels)

    assert set(metrics["split"]) == {TRAIN, VAL}
    assert len(metrics) == EXPECTED_SPLIT_METRIC_ROWS


def test_cluster_sizes_shares_sum_to_one_within_split():
    labels = build_labels_frame(
        np.arange(4),
        np.array([0, 0, 1, 1]),
        confidence=np.ones(4),
        gmm_conf=np.ones(4),
        split=np.array([TRAIN, VAL, TRAIN, TEST]),
    )

    sizes = cluster_sizes(labels)

    assert set(sizes["split"]) == {TRAIN, VAL, TEST}
    assert np.allclose(sizes.groupby("split")["share"].sum().to_numpy(), 1.0)
