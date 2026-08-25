import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from embedding_space_exploration.config import PRIMARY_SCALING, RANDOM_STATE


def residualize_embeddings(embeddings, covariates, *, fit_mask=None):
    """Regress named covariates out of every embedding dimension (OLS residuals).

    The **targeted** alternative to ``n_drop_components``. Dropping a leading PC
    removes the whole component -- and a PC whose R^2 with ``n_events`` is 0.38 is
    still 62% something else, which the drop discards too. Residualisation removes
    only the measured covariate *direction*: per dimension, fit ``dim ~ 1 +
    covariates`` on the ``train`` rows and keep the residual. The rank cost is the
    number of covariates, not the number of components.

    Both are arms to be measured, never assumed (Forooghi et al. 2024: isotropy can
    rise while usefulness falls) -- ``clustering/arms.py`` scores them side by side
    against the untouched baseline, and the arbiter stays downstream outcome
    separation.

    Note this removes only the *linear* projection onto the covariate span. A
    non-linear utilisation axis survives it, which is one reason the drop arm is
    kept alongside rather than replaced.

    Args:
        embeddings: Frame with ``person_id`` plus ``dim_0 .. dim_N`` columns.
        covariates: Frame with ``person_id`` plus one **numeric** column per
            covariate to remove (e.g. ``log_n_events``, ``obs_years``). Rows are
            aligned on ``person_id``; missing values are mean-imputed within the
            fit rows so a patient is never dropped from the embedding frame.
        fit_mask: Boolean array selecting the rows the regression is fit on
            (``train`` only, so held-out patients never inform the correction).
            Defaults to all rows.

    Returns:
        A copy of ``embeddings`` with the ``dim_*`` columns replaced by residuals.
    """
    person_id, matrix = _split(embeddings)
    if fit_mask is None:
        fit_mask = np.ones(len(matrix), dtype=bool)

    design = _design_matrix(covariates, person_id, fit_mask)
    coefficients, *_ = np.linalg.lstsq(design[fit_mask], matrix[fit_mask], rcond=None)

    residuals = matrix - design @ coefficients
    out = embeddings.copy()
    out[[c for c in embeddings.columns if c.startswith("dim_")]] = residuals
    return out


def prepare_matrix(
    embeddings,
    *,
    fit_mask=None,
    n_components=None,
    n_drop_components=None,
    scaling=PRIMARY_SCALING,
):
    """Turn a representation frame into a clustering-ready matrix.

    Three scalings, because the honest baseline, the anisotropic CLMBR embedding,
    and an interpretable clinical-variable table need different geometry:

    - ``"raw"`` (**baseline**): the model output unchanged -- full-dimensional
      Euclidean, **magnitude retained**, no L2, no PCA. The assumption-free space
      every geometric arm must beat. L2 discards vector *length* (a record-density
      / ``n_events`` proxy) on the hypothesis that phenotype lives in the *angle*;
      that hypothesis is exactly what the ``"spherical"`` arm has to earn against
      this baseline, so raw is the reference, not spherical.
    - ``"spherical"`` (**an arm to beat**): L2-normalise each vector (drop the
      length/activity axis) -> PCA fit on the ``train`` rows (decorrelate/denoise;
      optional ``n_drop_components`` "all-but-the-top" trick) -> re-L2-normalise, so
      K-means is cosine/spherical.
    - ``"standard"`` (**clinical variables / mixed-scale features**): z-score each
      column (fit on the ``train`` rows) and stop -- plain standardised Euclidean
      K-means. No L2 (row-normalising a [age, glucose, ...] vector is meaningless)
      and no PCA (the features are already low-dimensional and interpretable).

    Either way the fit (PCA basis or scaler statistics) uses only the ``fit_mask``
    rows, so held-out patients never inform the transform.

    Args:
        embeddings: Frame with ``person_id`` plus ``dim_0 .. dim_N`` columns.
        fit_mask: Boolean array selecting the ``train`` rows to fit on. Defaults
            to all rows.
        n_components: PCA components to keep (spherical only; default
            ``N_COMPONENTS``).
        n_drop_components: Leading PCA components to discard (spherical only;
            default ``N_DROP_COMPONENTS``).
        scaling: ``"raw"`` (baseline), ``"spherical"``, or ``"standard"`` (see
            above).

    Returns:
        ``(person_id array, float matrix)`` in input order.
    """
    person_id, matrix = _split(embeddings)
    if fit_mask is None:
        fit_mask = np.ones(len(matrix), dtype=bool)

    if scaling == "raw":
        # The baseline: no transform. K-means is translation-invariant, so leaving
        # the vectors un-centred is fine; nothing is fit, so fit_mask is unused.
        return person_id, matrix

    if scaling == "standard":
        scaler = StandardScaler().fit(matrix[fit_mask])
        return person_id, scaler.transform(matrix)

    n_components = N_COMPONENTS if n_components is None else n_components
    n_drop = N_DROP_COMPONENTS if n_drop_components is None else n_drop_components

    matrix = _l2_normalize(matrix)
    n_keep = _resolve_n_components(matrix[fit_mask], n_components, n_drop)
    pca = PCA(n_components=n_drop + n_keep, random_state=RANDOM_STATE)
    pca.fit(matrix[fit_mask])
    reduced = pca.transform(matrix)[:, n_drop:]
    return person_id, _l2_normalize(reduced)


# ======================================================================================
# HELPER FUNCTIONS AND CONSTANTS
# ======================================================================================

# Study Design Freeze section 9. PCA target dimensionality: enough to retain
# structure, low enough to escape high-dimensional distance concentration.
N_COMPONENTS = 50

# Study Design Freeze section 9. Leading PCA components to discard (anisotropy
# removal). 0 is the primary arm -- the honest baseline. drop2 and residualisation
# are declared *sensitivity arms*, measured and reported, never applied as fixes
# (Forooghi et al. 2024: isotropy can rise while usefulness falls).
N_DROP_COMPONENTS = 0


def _split(embeddings):
    """Split the embedding frame into (person_id array, float matrix)."""
    dims = [c for c in embeddings.columns if c.startswith("dim_")]
    person_id = embeddings["person_id"].to_numpy()
    return person_id, embeddings[dims].to_numpy(dtype="float64")


def _l2_normalize(matrix):
    """Scale each row to unit L2 norm (zero rows are left untouched)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _design_matrix(covariates, person_id, fit_mask):
    """Intercept + mean-imputed, fit-standardised covariate columns, in row order."""
    aligned = covariates.set_index("person_id").reindex(person_id)
    values = aligned[list(aligned.columns)].to_numpy(dtype="float64")

    # Impute and standardise using the fit rows only -- the same no-leakage rule the
    # PCA basis and the scaler statistics follow.
    fit_values = values[fit_mask]
    centre = np.nanmean(fit_values, axis=0)
    centre = np.where(np.isnan(centre), 0.0, centre)
    values = np.where(np.isnan(values), centre, values)

    scale = np.nanstd(values[fit_mask], axis=0)
    scale = np.where((scale == 0) | np.isnan(scale), 1.0, scale)

    return np.column_stack([np.ones(len(values)), (values - centre) / scale])


def _resolve_n_components(matrix, n_components, n_drop_components):
    """Clamp the kept-component count so PCA stays within matrix rank."""
    max_components = min(matrix.shape) - n_drop_components
    return max(1, min(n_components, max_components))
