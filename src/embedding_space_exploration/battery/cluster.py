import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from embedding_space_exploration.battery.prep import prepare_matrix
from embedding_space_exploration.config import (
    K_VALUES,
    N_REPEATS,
    PREDICTION_STRENGTH_THRESHOLD,
    PRIMARY_SCALING,
    RANDOM_STATE,
)
from embedding_space_exploration.splits import OPEN_SPLITS, TRAIN, split_label


def run_clustering(embeddings, split, *, scaling=PRIMARY_SCALING):
    """Derive clusters on the train split and assign every patient.

    The full clustering step, returning each output the task writes. The ``train``
    split drives every fit: the PCA basis, k selection (via train-internal
    prediction-strength resampling), and the K-means centroids. Every patient
    (including ``val`` and ``test``) is only *assigned* via those centroids.
    Cluster-distinctiveness metrics are reported for ``train`` and ``val`` only --
    ``test`` carries the reported numbers and is not inspected here (see
    ``splits``).

    Args:
        embeddings: Frame with ``person_id`` plus ``dim_0 .. dim_N`` columns.
        split: Frame mapping ``person_id`` to train/val/test.
        scaling: Preprocessing geometry (``prepare_matrix``): ``"spherical"`` for
            embeddings, ``"standard"`` for clinical-variable / mixed-scale features.

    Returns:
        Mapping of output name to DataFrame: ``labels``, ``prediction_strength``,
        ``metrics`` and ``sizes``.
    """
    labels_by_row = split_label(embeddings["person_id"], split)
    train_mask = labels_by_row == TRAIN

    person_id, matrix = prepare_matrix(embeddings, fit_mask=train_mask, scaling=scaling)
    sweep = prediction_strength_sweep(matrix[train_mask])
    k = choose_k(sweep)
    model = fit_clusters(matrix[train_mask], k)

    labels = build_labels_frame(
        person_id,
        model.predict(matrix),
        confidence=assignment_confidence(matrix, model),
        gmm_conf=gmm_confidence(matrix, k, fit_matrix=matrix[train_mask]),
        split=labels_by_row,
    )
    metrics = cluster_metrics_by_split(
        matrix, labels["cluster"].to_numpy(), labels_by_row
    )
    metrics.insert(0, "chosen_k", k)
    metrics["prediction_strength_threshold"] = PREDICTION_STRENGTH_THRESHOLD

    return {
        "labels": labels,
        "prediction_strength": sweep,
        "metrics": metrics,
        "sizes": cluster_sizes(labels),
    }


def prediction_strength_sweep(
    matrix, *, k_values=None, n_repeats=None, random_state=None
):
    """Prediction-strength curve over candidate cluster counts.

    Prediction strength (Tibshirani & Walther 2005) measures how reproducibly a
    clustering generalises: split the data, cluster each half, and check whether
    pairs grouped together in the test half are still grouped together when
    assigned to the *train* half's centroids. This is the selection method used
    by both reference papers (Lian et al. 2026, Fan et al. 2025).

    Args:
        matrix: Prepared (row-normalised, reduced) float matrix.
        k_values: Candidate cluster counts (default ``K_VALUES``).
        n_repeats: Random train/test splits to average over (default
            ``N_REPEATS``).
        random_state: Seed for the split/fit RNG (default ``RANDOM_STATE``).

    Returns:
        One row per k: ``k``, ``prediction_strength_mean``,
        ``prediction_strength_std`` (across repeats).
    """
    k_values = K_VALUES if k_values is None else k_values
    n_repeats = N_REPEATS if n_repeats is None else n_repeats
    random_state = RANDOM_STATE if random_state is None else random_state

    rng = np.random.default_rng(random_state)
    rows = [
        {
            "k": k,
            **_summarise(
                [_prediction_strength(matrix, k, rng) for _ in range(n_repeats)]
            ),
        }
        for k in k_values
    ]
    return pd.DataFrame(rows)


def choose_k(sweep, *, threshold=None):
    """Pick k as the largest count whose mean prediction strength clears a bar.

    Following Tibshirani & Walther, prefer the *largest* well-supported k. If no
    k clears the threshold, fall back to the k with the highest mean strength so
    the pipeline still produces a labelling (flagged via the metrics output).

    Args:
        sweep: Output of ``prediction_strength_sweep``.
        threshold: Minimum acceptable mean prediction strength (default
            ``PREDICTION_STRENGTH_THRESHOLD``).

    Returns:
        The chosen number of clusters (int).
    """
    threshold = PREDICTION_STRENGTH_THRESHOLD if threshold is None else threshold
    passing = sweep[sweep["prediction_strength_mean"] >= threshold]
    if passing.empty:
        return int(sweep.loc[sweep["prediction_strength_mean"].idxmax(), "k"])
    return int(passing["k"].max())


def fit_clusters(matrix, k, *, random_state=None):
    """Fit spherical K-means (K-means on the row-normalised matrix).

    Args:
        matrix: Prepared float matrix.
        k: Number of clusters.
        random_state: Seed (default ``RANDOM_STATE``).

    Returns:
        The fitted ``KMeans`` model (``.labels_`` holds the assignments).
    """
    random_state = RANDOM_STATE if random_state is None else random_state
    return KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=random_state).fit(
        matrix
    )


def assignment_confidence(matrix, model):
    """Per-patient hard-assignment confidence from K-means distances.

    A softmax over negative squared distances to the centroids; the returned
    value is the probability mass on the assigned cluster. ~1 means the patient
    sits firmly in one cluster, ~1/k means it is on a boundary. A heuristic
    (K-means has no probability model), useful for flagging ambiguous patients.

    Args:
        matrix: Prepared float matrix.
        model: A fitted ``KMeans`` model.

    Returns:
        1-D array of confidences in ``[0, 1]``, one per row of ``matrix``.
    """
    distances = model.transform(matrix)
    logits = -(distances**2)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.max(axis=1)


def gmm_confidence(matrix, k, *, fit_matrix=None, random_state=None):
    """Soft-assignment confidence from a Gaussian mixture probe.

    A second lens on membership certainty (Lian et al. report assignment
    confidence distributions). Diagonal covariance is used: it is robust in the
    reduced space and avoids singular full-covariance fits on small cohorts.

    Args:
        matrix: Prepared float matrix to score (all patients).
        k: Number of mixture components (matched to the K-means k).
        fit_matrix: Rows to fit the mixture on (the train patients). Defaults to
            ``matrix`` (fit and score on the same rows).
        random_state: Seed (default ``RANDOM_STATE``).

    Returns:
        1-D array: the top posterior probability per patient.
    """
    random_state = RANDOM_STATE if random_state is None else random_state
    fit_matrix = matrix if fit_matrix is None else fit_matrix
    gmm = GaussianMixture(
        n_components=k, covariance_type="diag", random_state=random_state
    ).fit(fit_matrix)
    return gmm.predict_proba(matrix).max(axis=1)


def cluster_metrics(matrix, labels):
    """Internal cluster-distinctiveness scores (descriptive, not proof).

    Silhouette and Calinski-Harabasz are the two metrics reported by Fan et al.;
    Davies-Bouldin is added as a third (lower is better). These describe
    separation in the embedding space and should be read alongside prediction
    strength and downstream outcome separation, not on their own.

    Args:
        matrix: Prepared float matrix.
        labels: Cluster assignment per row.

    Returns:
        A single-row DataFrame of the three scores.
    """
    return pd.DataFrame(
        [
            {
                "n_clusters": len(np.unique(labels)),
                "silhouette": float(silhouette_score(matrix, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
                "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
            }
        ]
    )


def cluster_metrics_by_split(matrix, labels, split_labels):
    """Cluster-distinctiveness metrics for the train and val splits.

    Train and val rows are scored independently so the val numbers reflect held-out
    patients assigned to the train-fit centroids. **``test`` is deliberately
    excluded** -- it carries the reported numbers and is not inspected during
    development (``splits.OPEN_SPLITS``). A split is skipped if it has no rows or
    fewer than two clusters present (silhouette undefined).

    Args:
        matrix: Prepared float matrix (all patients).
        labels: Cluster assignment per row.
        split_labels: Array of train/val/test per row, aligned with ``matrix``.

    Returns:
        DataFrame of per-split metrics with a leading ``split`` column (train/val).
    """
    frames = []
    for name in OPEN_SPLITS:
        mask = split_labels == name
        if (
            mask.sum() == NO_ROWS_IN_SPLIT
            or len(np.unique(labels[mask])) < MIN_CLUSTERS_FOR_METRICS
        ):
            continue
        metrics = cluster_metrics(matrix[mask], labels[mask])
        metrics.insert(0, "split", name)
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def cluster_sizes(labels_frame):
    """Patient count and within-split share per cluster, largest first."""
    sizes = (
        labels_frame.groupby(["split", "cluster"])
        .size()
        .rename("n_patients")
        .reset_index()
    )
    sizes["share"] = sizes["n_patients"] / sizes.groupby("split")[
        "n_patients"
    ].transform("sum")
    return sizes.sort_values(
        ["split", "n_patients"], ascending=[True, False]
    ).reset_index(drop=True)


def build_labels_frame(person_id, labels, *, confidence, gmm_conf, split):
    """Assemble the per-patient cluster-label table."""
    return pd.DataFrame(
        {
            "person_id": person_id,
            "cluster": labels,
            "split": split,
            "assignment_confidence": confidence,
            "gmm_confidence": gmm_conf,
        }
    )


# ======================================================================================
# HELPER FUNCTIONS AND CONSTANTS
# ======================================================================================

KMEANS_N_INIT = 10
NO_ROWS_IN_SPLIT = 0
MIN_CLUSTERS_FOR_METRICS = 2


def _prediction_strength(matrix, k, rng):
    """One train/test split's prediction strength for a given k."""
    n = len(matrix)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[: n // 2], perm[n // 2 :]
    seed = int(rng.integers(np.iinfo(np.int32).max))

    train_model = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed).fit(
        matrix[train_idx]
    )
    test_labels = KMeans(
        n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed
    ).fit_predict(matrix[test_idx])
    predicted = train_model.predict(matrix[test_idx])
    return _comembership_strength(test_labels, predicted)


def _comembership_strength(test_labels, predicted):
    """Min over test clusters of the within-cluster co-membership rate.

    For each cluster found in the test set, the fraction of its (ordered) point
    pairs that the train model also assigns to the same cluster.
    """
    strengths = []
    for cluster in np.unique(test_labels):
        members = np.where(test_labels == cluster)[0]
        counts = np.bincount(predicted[members])
        same_pairs = int(np.sum(counts * (counts - 1)))
        total_pairs = len(members) * (len(members) - 1)
        strengths.append(1.0 if total_pairs == 0 else same_pairs / total_pairs)
    return float(min(strengths))


def _summarise(scores):
    """Mean/std of a list of prediction-strength scores."""
    return {
        "prediction_strength_mean": float(np.mean(scores)),
        "prediction_strength_std": float(np.std(scores)),
    }
