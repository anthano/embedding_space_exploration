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
    NULL_MARGIN_THRESHOLD,
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


def null_margin(tendency):
    """Headroom-normalised distance from the real sweep to its own null band.

    ``exceeds_null`` is a boolean and throws the magnitude away: at ``k=10`` a
    real 0.170 against a null median 0.153 counts the same as a real 0.769
    against 0.389 at ``k=4``, though one is 2% above chance and the other is
    nearly double it.

    Prediction strength decays with k for real *and* null data, so no absolute
    bar is comparable across k -- 0.8 is easy at k=2 and unreachable by k=8. Two
    obvious normalisations are also k-biased: ``real - null`` favours low k,
    where the whole scale is larger, and ``real / null`` favours high k, where
    the ceiling ``1 / null`` is larger (1.7 at k=2, 6.5 at k=10).

    So: the share of the **available headroom above the null** that the space
    actually captures, ``(real - null_median) / (1 - null_median)``. Prediction
    strength is bounded by 1, so this is bounded by 1 at every k -- comparable
    across k, and across spaces, which is what an 18-way comparison needs.

    Args:
        tendency: Output of ``cluster_tendency_vs_null``.

    Returns:
        ``tendency`` with a ``headroom_margin`` column added.
    """
    out = tendency.copy()
    out["headroom_margin"] = (
        (out["prediction_strength"] - out["null_ps_median"])
        / (1 - out["null_ps_median"])
    ).round(4)
    return out


def verdict_label(margin, *, beats_null, threshold=NULL_MARGIN_THRESHOLD):
    """The gate's three-way label, as a function of the recorded margin alone.

    Split out from ``null_gate_verdict`` so that a *finished* run can be re-read
    under a different threshold. The margin is the continuous statistic and does
    not go stale; the label is the only part that depends on the constant. Any
    consumer that reports a verdict should derive it here rather than read a
    stored label, which would freeze ``NULL_MARGIN_THRESHOLD`` at whatever value
    happened to be set when that run was executed -- and so silently keep
    reporting the old bar after the constant is revised.

    Args:
        margin: ``max_headroom_margin`` for the space.
        beats_null: Whether any k in the sweep exceeded the null band.
        threshold: Minimum headroom margin for structure to be believed.

    Returns:
        One of ``"CONTINUOUS"``, ``"WEAK"``, ``"DISCRETE"``.
    """
    if not beats_null:
        return "CONTINUOUS"
    return "WEAK" if margin < threshold else "DISCRETE"


def null_gate_verdict(tendency, *, threshold=NULL_MARGIN_THRESHOLD):
    """Reduce the per-k table to a single go / no-go read on discrete structure.

    **This deliberately does not select k.** The gate's job is whether a space
    has discrete structure at all; ``cluster.choose_k`` is a separate question,
    and for a comparison across spaces k should be *fixed and declared* rather
    than chosen per space -- a space clustered at k=4 and one at k=7 have
    silhouettes that cannot be compared.

    The rule this replaces took the *largest* k that beat the null and cleared an
    absolute prediction-strength bar. That fails in a specific, reproducible way:
    because ``exceeds_null`` is close to automatic in the high-k tail, "largest"
    resolves to the weakest evidence available, and lowering the bar to admit a
    genuine k admits several spurious larger ones first.

    Note the verdict is *derived* from ``max_headroom_margin``, which is recorded
    alongside it. Re-reading the same run under a different threshold costs
    nothing.

    Args:
        tendency: Output of ``cluster_tendency_vs_null``.
        threshold: Minimum headroom margin for structure to be believed.

    Returns:
        A single-row DataFrame: ``max_headroom_margin``, ``k_at_max_margin``
        (a diagnostic, **not** a selected k), ``any_k_beats_null``, ``verdict``.
    """
    scored = null_margin(tendency)
    best = scored.loc[scored["headroom_margin"].idxmax()]
    beats_null = bool(scored["exceeds_null"].any())
    margin = float(best["headroom_margin"])
    label = verdict_label(margin, beats_null=beats_null, threshold=threshold)

    if label == "CONTINUOUS":
        verdict = (
            "CONTINUOUS: no k beats a single covariance-matched Gaussian null -- no "
            "discrete cluster structure (a real result, not a failure)."
        )
    elif label == "WEAK":
        verdict = (
            f"WEAK: the best margin over the null is {margin:.3f}, below {threshold} "
            "-- more clusterable than noise, but not by much."
        )
    else:
        verdict = (
            f"DISCRETE: captures {margin:.3f} of the headroom above a "
            "covariance-matched null."
        )

    return pd.DataFrame(
        [
            {
                "max_headroom_margin": margin,
                "k_at_max_margin": int(best["k"]),
                "any_k_beats_null": beats_null,
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
