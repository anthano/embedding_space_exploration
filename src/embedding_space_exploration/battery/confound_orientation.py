"""Confound-orientation (PCR): which nuisance axes the embedding's variance encodes.

Principal-component regression, in the single-cell-foundation-model style
(Kedzierska et al. 2025): rather than waiting for cluster labels and cross-tabbing
them post-hoc, we interrogate the **space itself** -- for each principal component,
how much of that PC's variance is explained by each nuisance / anchor variable
(`n_events`, age, gender, race, and whatever label the dataset declares). This
diagnoses *whether the leading PC(s)
are the `n_events` / site axis*, which is exactly the evidence `N_DROP_COMPONENTS`
needs to be a targeted drop rather than a blind one.

**Swept over the space set** (like the null gate):

- On **raw** (no L2) it asks *"is magnitude the `n_events` axis -- does L2 even earn
  its place?"*. If raw's PC0 is dominated by `n_events`, that is the direct evidence
  for L2; if not, L2 is discarding something else.
- On the **L2** PCA (set ``l2_normalize=True`` -- the exact PCA `N_DROP` acts on) it
  asks *"is the top post-L2 PC still artifact -- does `N_DROP` help?"*.

> Guardrail (Forooghi et al. 2024): variance-explained is **not** harm. A nuisance
> can load on a PC that also carries phenotype, so PCR *informs* the drop decision;
> it never licenses stripping variance just because it is inconvenient. The arbiter
> stays downstream outcome separation.

Pure functions only, and dataset-agnostic: the covariate *assembler* is
per-dataset and lives with the data layer. This module only needs a frame of
``person_id`` plus one column per nuisance/anchor variable.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from embedding_space_exploration.config import RANDOM_STATE

# Leading PCs written to the per-PC table (the ones N_DROP could plausibly drop).
# The per-covariate variance-weighted sum still uses *all* components.
N_TOP_PCS = 20

_MIN_POINTS_FOR_R2 = 2


def principal_component_regression(
    embeddings,
    covariates,
    *,
    fit_mask=None,
    n_top_pcs=N_TOP_PCS,
    l2_normalize=False,
):
    """Per-PC and per-covariate association between embedding variance and nuisances.

    Args:
        embeddings: Frame with ``person_id`` plus ``dim_0 .. dim_N`` columns.
        covariates: Frame with ``person_id`` plus one column per nuisance/anchor.
            Numeric columns are treated as continuous (R^2 = squared correlation);
            non-numeric as categorical (R^2 = correlation ratio eta^2).
        fit_mask: Boolean array selecting the ``train`` rows to fit the PCA on and
            compute associations over (no leakage). Defaults to all rows.
        n_top_pcs: Number of leading PCs to write to the per-PC table.
        l2_normalize: L2-normalise rows before the PCA -- set for the **L2**
            (spherical) arm so the PCA is the one ``N_DROP`` acts on; leave ``False``
            for the **raw** baseline.

    Returns:
        ``{"per_pc": DataFrame, "per_covariate": DataFrame}``. ``per_pc`` has one row
        per leading PC (``pc``, ``explained_variance_ratio``, ``r2__{cov}`` per
        covariate, ``dominant_covariate``, ``dominant_r2``). ``per_covariate`` has
        one row per covariate (``variance_weighted_r2`` = sum of evr*R^2 over all
        PCs = fraction of total embedding variance associated with it,
        ``leading_pc_r2`` = R^2 with PC0, ``top_pc``, ``top_pc_r2``), most-associated
        first.
    """
    dims = [c for c in embeddings.columns if c.startswith("dim_")]
    person_id = embeddings["person_id"].to_numpy()
    matrix = embeddings[dims].to_numpy(dtype="float64")
    if fit_mask is None:
        fit_mask = np.ones(len(matrix), dtype=bool)
    if l2_normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)

    pca = PCA(random_state=RANDOM_STATE).fit(matrix[fit_mask])
    scores = pca.transform(matrix[fit_mask])
    evr = pca.explained_variance_ratio_

    cov = covariates.set_index("person_id").reindex(person_id[fit_mask])
    cov_cols = list(cov.columns)

    r2 = np.array(
        [[_r2(scores[:, i], cov[c]) for c in cov_cols] for i in range(scores.shape[1])]
    )

    return {
        "per_pc": _per_pc_frame(evr, r2, cov_cols, n_top_pcs),
        "per_covariate": _per_covariate_frame(evr, r2, cov_cols),
    }


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _r2(pc_scores, covariate):
    """R^2 of a single PC against one covariate (correlation^2 or eta^2)."""
    y = np.asarray(pc_scores, dtype="float64")
    if pd.api.types.is_numeric_dtype(covariate):
        return _numeric_r2(y, covariate.to_numpy(dtype="float64"))
    return _categorical_eta_squared(y, covariate)


def _numeric_r2(y, x):
    """Squared Pearson correlation, NaN-safe and constant-safe."""
    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < _MIN_POINTS_FOR_R2:
        return np.nan
    if np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return 0.0
    return float(np.corrcoef(x[valid], y[valid])[0, 1] ** 2)


def _categorical_eta_squared(y, covariate):
    """Correlation ratio eta^2: share of PC variance explained by the categories."""
    valid = covariate.notna().to_numpy() & ~np.isnan(y)
    yv = y[valid]
    groups = covariate.to_numpy()[valid]
    if len(yv) < _MIN_POINTS_FOR_R2:
        return np.nan
    grand = yv.mean()
    ss_total = float(((yv - grand) ** 2).sum())
    if ss_total == 0:
        return 0.0
    ss_between = sum(
        len(yv[groups == level]) * (yv[groups == level].mean() - grand) ** 2
        for level in np.unique(groups)
    )
    return float(ss_between / ss_total)


def _per_pc_frame(evr, r2, cov_cols, n_top_pcs):
    """One row per leading PC: evr, R^2 per covariate, and the dominant covariate."""
    rows = []
    for i in range(min(n_top_pcs, r2.shape[0])):
        row = {"pc": i, "explained_variance_ratio": _round(evr[i])}
        row.update({f"r2__{c}": _round(r2[i, j]) for j, c in enumerate(cov_cols)})
        best = int(np.nanargmax(r2[i]))
        row["dominant_covariate"] = cov_cols[best]
        row["dominant_r2"] = _round(r2[i, best])
        rows.append(row)
    return pd.DataFrame(rows)


def _per_covariate_frame(evr, r2, cov_cols):
    """One row per covariate: variance-weighted association (all PCs) + its top PC."""
    rows = []
    for j, c in enumerate(cov_cols):
        column = np.nan_to_num(r2[:, j])
        top_pc = int(np.nanargmax(r2[:, j]))
        rows.append(
            {
                "covariate": c,
                "variance_weighted_r2": _round(float((evr * column).sum())),
                "leading_pc_r2": _round(r2[0, j]),
                "top_pc": top_pc,
                "top_pc_r2": _round(r2[top_pc, j]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("variance_weighted_r2", ascending=False)
        .reset_index(drop=True)
    )


def _round(value, ndigits=4):
    """Round a possibly-NaN float for stable CSV output."""
    return value if value is None or np.isnan(value) else round(float(value), ndigits)
