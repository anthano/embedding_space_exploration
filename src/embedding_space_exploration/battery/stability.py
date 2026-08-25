"""Partition stability: is the exploratory k-means partition reproducible?

The second stability layer beside Tibshirani-Walther prediction strength (the brief's
deferred Monti/consensus item), and one both reference papers report (Fan et al. 2025,
Lian et al. 2026). For each candidate k we cluster the full data (the reference
partition), then repeatedly **bootstrap** the patients, re-cluster, label every patient
with the bootstrap centroids, and measure the **Adjusted Rand Index** against the
reference. High, tight ARI = the same partition keeps re-emerging under resampling; low
or variable ARI = the k-means solution is an artefact of the particular sample.

This is an *exploration* metric -- "is there a partition that reproduces?" -- not a
claim that the partition is clinically meaningful (that stays with outcome separation).
It complements prediction strength: prediction strength splits once and checks pair
co-membership; bootstrap ARI resamples with replacement and checks whole-partition
agreement, so agreement across both is stronger evidence than either alone.

Pure functions only; I/O + pytask wiring live in ``task_stability``.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from embedding_space_exploration.battery.cluster import KMEANS_N_INIT
from embedding_space_exploration.config import K_VALUES, RANDOM_STATE

# Study Design Freeze section 9. Bootstrap resamples per k. 25 gives a usable
# mean/std at tolerable cost.
N_BOOTSTRAP_PARTITION = 25

_SEED_MAX = np.iinfo(np.int32).max


def stability_sweep(
    matrix, *, k_values=None, n_boot=N_BOOTSTRAP_PARTITION, random_state=RANDOM_STATE
):
    """Bootstrap Adjusted Rand Index per k against the full-data reference partition.

    Args:
        matrix: Prepared ``train`` matrix (the space clustering would run on).
        k_values: Candidate cluster counts (default ``K_VALUES``, the pipeline's).
        n_boot: Bootstrap resamples per k.
        random_state: Seed for the reference fit and the resampling.

    Returns:
        One row per k: ``k``, ``mean_ari``, ``std_ari`` across the bootstraps.
    """
    k_values = K_VALUES if k_values is None else k_values
    rng = np.random.default_rng(random_state)
    n = len(matrix)

    rows = []
    for k in k_values:
        reference = KMeans(
            n_clusters=k, n_init=KMEANS_N_INIT, random_state=random_state
        ).fit_predict(matrix)
        aris = [_bootstrap_ari(matrix, k, reference, n, rng) for _ in range(n_boot)]
        rows.append(
            {
                "k": int(k),
                "mean_ari": round(float(np.mean(aris)), 4),
                "std_ari": round(float(np.std(aris)), 4),
            }
        )
    return pd.DataFrame(rows)


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _bootstrap_ari(matrix, k, reference, n, rng):
    """One bootstrap: resample, re-cluster, label all patients, ARI vs reference."""
    idx = rng.integers(n, size=n)  # sample with replacement
    seed = int(rng.integers(_SEED_MAX))
    model = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed).fit(
        matrix[idx]
    )
    return adjusted_rand_score(reference, model.predict(matrix))
