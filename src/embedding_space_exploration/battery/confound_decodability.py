"""Nuisance decodability: what a probe can pull out of a space, not what it points at.

C1 (``confound_orientation``) asks whether nuisance variance is *aligned* with the
leading principal components. That is the right question for clustering, because
K-means partitions along the directions carrying the most variance -- orientation
is what corrupts a partition. This module asks a different one: can a *probe*
recover the nuisance at all?

The two dissociate, and the dissociation is the point. A confound written along a
trailing direction, or written non-linearly across several, contributes almost
nothing to any principal component -- C1 reads ~0 -- while a learner recovers it
perfectly. That space is clean for clustering and dirty for every downstream
probe, and reporting only C1 would license the wrong claim about it.

Three readings of the same covariate, in increasing strength:

- PCR variance-weighted R^2 (C1, in ``confound_orientation``): is it *oriented*?
- linear-probe R^2 (here): is it *linearly decodable*, wherever it lives?
- nonlinear-probe R^2 (here): is it decodable **at all**?

Fit on ``train``, scored on ``val`` -- never ``test``, which carries the reported
numbers (see ``splits``). R^2 is reported unclipped, so a negative value means the
probe does worse than predicting the mean; read that as "not decodable", not as a
magnitude.

Numeric covariates only. A categorical nuisance (site, gender) needs the
classifier variant with AUROC in place of R^2, and arrives with the first dataset
that has one.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from embedding_space_exploration.config import RANDOM_STATE
from embedding_space_exploration.data_management.splits import TRAIN, VAL, split_label

# Ridge over a wide alpha grid rather than plain OLS: the embedding can be wider
# than the train split (768 dims, ~1.2k rows), where an unregularised fit
# interpolates the training rows and reports a linear R^2 that is an artefact of
# the dimension rather than a property of the space.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0)

# Below this many usable rows on either side the probe is not reported at all --
# an R^2 from a handful of rows is noise wearing a number.
_MIN_ROWS = 30


def confound_decodability(embeddings, covariates, split, *, random_state=RANDOM_STATE):
    """Linear and nonlinear recoverability of each numeric nuisance from the space.

    Args:
        embeddings: Frame with ``person_id`` plus ``dim_0 .. dim_N`` columns.
        covariates: Frame with ``person_id`` plus one column per nuisance.
            Non-numeric columns are skipped (see the module docstring).
        split: Frame mapping ``person_id`` to train/val/test. The probe is fit on
            ``train`` and scored on ``val``.
        random_state: Seed for the nonlinear learner.

    Returns:
        One row per numeric covariate: ``covariate``, ``n_fit``, ``n_score``,
        ``r2_linear``, ``r2_nonlinear``, and ``nonlinear_gain``
        (``r2_nonlinear - r2_linear``) -- the part of the nuisance that no linear
        probe, and so no principal component, can reach. Most decodable first.
    """
    person_id = embeddings["person_id"]
    dims = [c for c in embeddings.columns if c.startswith("dim_")]
    matrix = embeddings[dims].to_numpy(dtype="float64")

    labels = split_label(person_id, split)
    train, val = labels == TRAIN, labels == VAL
    aligned = covariates.set_index("person_id").reindex(person_id.to_numpy())

    rows = [
        _probe_covariate(matrix, aligned[name], name, train, val, random_state)
        for name in aligned.columns
        if pd.api.types.is_numeric_dtype(aligned[name])
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("r2_nonlinear", ascending=False)
        .reset_index(drop=True)
    )


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _probe_covariate(matrix, covariate, name, train, val, random_state):
    """Fit both probes on the train rows and score them on the held-out val rows."""
    y = covariate.to_numpy(dtype="float64")
    usable = ~np.isnan(y)
    fit, score = train & usable, val & usable

    row = {
        "covariate": name,
        "n_fit": int(fit.sum()),
        "n_score": int(score.sum()),
        "r2_linear": np.nan,
        "r2_nonlinear": np.nan,
        "nonlinear_gain": np.nan,
    }
    if fit.sum() < _MIN_ROWS or score.sum() < _MIN_ROWS:
        return row

    linear = RidgeCV(alphas=RIDGE_ALPHAS).fit(matrix[fit], y[fit])
    nonlinear = HistGradientBoostingRegressor(random_state=random_state).fit(
        matrix[fit], y[fit]
    )
    row["r2_linear"] = round(
        float(r2_score(y[score], linear.predict(matrix[score]))), 4
    )
    row["r2_nonlinear"] = round(
        float(r2_score(y[score], nonlinear.predict(matrix[score]))), 4
    )
    row["nonlinear_gain"] = round(row["r2_nonlinear"] - row["r2_linear"], 4)
    return row


def _r2(observed, predicted):
    """Held-out R^2, rounded for stable output."""
    return round(float(r2_score(observed, predicted)), 4)
