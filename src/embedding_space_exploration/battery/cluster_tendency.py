"""Cluster tendency vs a covariance-matched null -- the gate for Step 2.

The check nobody in this literature runs, and the antidote to the Corpas trap:
prediction strength (and every internal metric) will happily report a "good" k>=2
even for a single anisotropic blob with no discrete structure. So before believing
any partition we ask -- *is the real space more clusterable than unstructured data
of the same shape?*

Method (Dinga et al. 2019, biotype non-replication): fit **one** multivariate
Gaussian to the prepared ``train`` matrix (its mean + full covariance -- a
deliberately "dumb", unimodal null), draw synthetic datasets of the same size, and
run the **identical** ``prediction_strength_sweep`` on each. The spread of null
prediction strengths per k is the band; only real structure that **clears the null
band** is believed.

This gates ``clustering`` (Step 2). It can legitimately return *"continuous, not
discrete"* -- if no k beats the null, that is a real result for the
categorical-vs-dimensional debate in psychosis (Pillinger's *variance* story admits
a spectrum, not necessarily clusters), not a pipeline failure.

**The null shape is space-dependent** (``resphere_null``). On the **raw** Euclidean
baseline the covariance-matched Gaussian is Dinga's null exactly as designed -- no
constraint on the data, so a Gaussian is the right unstructured comparator. On an
**L2-spherical** arm the data lives on a unit shell (``||x|| = 1``) while a Gaussian
fills the ball, a manifold mismatch that would rig the comparison; re-projecting the
Gaussian draws back onto the sphere (``resphere_null=True``) puts real and null on
the same shell. (A single von Mises-Fisher is the more principled spherical null; the
re-projected Gaussian is the cheap, mean+cov-matched stand-in.)

Pure functions only; I/O + pytask wiring live in ``task_cluster_tendency``.

Note the cost: ``n_draws`` null datasets, each a full prediction-strength sweep
(two K-means fits per k per repeat), so this is the most expensive battery member.
It runs once, on ``train``, as a gate.
"""

import numpy as np
import pandas as pd

from embedding_space_exploration.battery.cluster import prediction_strength_sweep
from embedding_space_exploration.config import (
    K_VALUES,
    N_REPEATS,
    PREDICTION_STRENGTH_THRESHOLD,
    RANDOM_STATE,
)

# Study Design Freeze section 9. Null datasets drawn from the covariance-matched
# Gaussian. More draws = a smoother band; 20 gives a usable p95 at a tolerable cost
# for a one-time gate.
N_NULL_DRAWS = 20

# Study Design Freeze section 9. The real sweep must clear this percentile of the
# null band at a given k for that k's structure to be believed (a one-sided "better
# than 95% of null draws" test).
NULL_UPPER_PERCENTILE = 95

_SEED_MAX = np.iinfo(np.int32).max
# Ridge added to the null covariance so the L2-normalised (rank-deficient) prepared
# matrix still yields a positive-definite Gaussian to sample from.
_COV_RIDGE = 1e-9


def cluster_tendency_vs_null(
    matrix,
    *,
    n_draws=N_NULL_DRAWS,
    k_values=None,
    n_repeats=None,
    random_state=RANDOM_STATE,
    resphere_null=False,
):
    """Per-k real prediction strength against a covariance-matched null band.

    Args:
        matrix: The ``train`` matrix ``cluster`` would sweep for this space --
            raw or a prepared arm -- so the comparison is apples-to-apples.
        n_draws: Null datasets to draw from the matched Gaussian (default
            ``N_NULL_DRAWS``).
        k_values: Candidate cluster counts (default ``K_VALUES``, the pipeline's).
        n_repeats: Train/test resamples per k (default ``N_REPEATS``, the
            pipeline's).
        random_state: Seed for both the real sweep and the null draws (default
            ``RANDOM_STATE`` -- matches ``run_clustering``).
        resphere_null: L2-normalise the Gaussian null draws back onto the unit
            sphere. Set for an **L2-spherical** arm (so real and null share the
            shell); leave ``False`` for the **raw** Euclidean baseline (a plain
            Gaussian is the correct Dinga null there).

    Returns:
        One row per k: ``k``, ``prediction_strength`` (real), ``null_ps_median``,
        ``null_ps_p95``, ``exceeds_null`` (real > the null p95 at that k).
    """
    k_values = K_VALUES if k_values is None else k_values
    n_repeats = N_REPEATS if n_repeats is None else n_repeats

    real = prediction_strength_sweep(
        matrix, k_values=k_values, n_repeats=n_repeats, random_state=random_state
    ).set_index("k")["prediction_strength_mean"]

    null = _null_curves(
        matrix,
        n_draws=n_draws,
        k_values=k_values,
        n_repeats=n_repeats,
        random_state=random_state,
        resphere_null=resphere_null,
    )
    null_median = null.median(axis=1)
    null_p95 = null.quantile(NULL_UPPER_PERCENTILE / 100.0, axis=1)

    rows = [
        {
            "k": int(k),
            "prediction_strength": _round(real.loc[k]),
            "null_ps_median": _round(null_median.loc[k]),
            "null_ps_p95": _round(null_p95.loc[k]),
            "exceeds_null": bool(real.loc[k] > null_p95.loc[k]),
        }
        for k in k_values
    ]
    return pd.DataFrame(rows)


def null_gate_verdict(tendency, *, threshold=PREDICTION_STRENGTH_THRESHOLD):
    """Reduce the per-k table to a single go / no-go read for Step 2.

    A k is "real" only if it both **beats the null band** and clears the pipeline's
    prediction-strength ``threshold`` (stable *and* better-than-noise). If no k
    qualifies, the honest verdict is that the space is continuous, not discrete.

    Args:
        tendency: Output of ``cluster_tendency_vs_null``.
        threshold: The prediction-strength bar (default the pipeline's).

    Returns:
        A single-row DataFrame: ``any_k_beats_null``,
        ``largest_k_beats_null_and_threshold`` (NaN if none), ``verdict``.
    """
    beats = tendency[tendency["exceeds_null"]]
    strong = beats[beats["prediction_strength"] >= threshold]
    largest = int(strong["k"].max()) if not strong.empty else None

    if largest is not None:
        verdict = (
            f"DISCRETE: structure at k<={largest} beats the covariance-matched null "
            f"and clears prediction-strength {threshold}."
        )
    elif not beats.empty:
        verdict = (
            "WEAK: some k beat the null but none also clears the prediction-strength "
            f"threshold {threshold} -- treat as fragile, not confirmed."
        )
    else:
        verdict = (
            "CONTINUOUS: no k beats a single covariance-matched Gaussian null -- no "
            "discrete cluster structure (a real result, not a failure)."
        )

    return pd.DataFrame(
        [
            {
                "any_k_beats_null": bool(not beats.empty),
                "largest_k_beats_null_and_threshold": (
                    largest if largest is not None else np.nan
                ),
                "verdict": verdict,
            }
        ]
    )


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _null_curves(matrix, *, n_draws, k_values, n_repeats, random_state, resphere_null):
    """Prediction-strength curves for ``n_draws`` covariance-matched null datasets.

    Returns a DataFrame indexed by k, one column per null draw.
    """
    rng = np.random.default_rng(random_state)
    mean = matrix.mean(axis=0)
    cov = np.cov(matrix, rowvar=False) + _COV_RIDGE * np.eye(matrix.shape[1])
    n = len(matrix)

    curves = {}
    for draw in range(n_draws):
        # check_valid="ignore": a rank-deficient covariance (n < d, or the L2
        # constraint) samples within the data's own subspace, which is the desired
        # support-matched null, not an error.
        sample = rng.multivariate_normal(mean, cov, size=n, check_valid="ignore")
        if resphere_null:
            norms = np.linalg.norm(sample, axis=1, keepdims=True)
            sample = sample / np.maximum(norms, 1e-12)
        seed = int(rng.integers(_SEED_MAX))
        curves[draw] = prediction_strength_sweep(
            sample, k_values=k_values, n_repeats=n_repeats, random_state=seed
        ).set_index("k")["prediction_strength_mean"]
    return pd.DataFrame(curves)


def _round(value, ndigits=4):
    """Round a float for stable, readable CSV output."""
    return round(float(value), ndigits)
