"""Score the battery against planted ground truth -- one synthetic space at a time.

The instrument being calibrated is the battery; this is the ruler held against it.
For one cell and one preprocessing arm it runs every applicable check, then adds
the columns only a simulation can supply: what the check *reported* next to what
was *planted*.

Two kinds of check, scored differently, and the distinction matters more than any
individual number:

- **With a truth counterpart** -- cluster ARI against the planted labels, RankMe
  against ``intrinsic_dim``, mean-cosine against ``anisotropy``. These give a
  measured-minus-declared error, and an error curve across a sweep is a
  calibration in the ordinary sense.
- **Without one** -- prediction strength, the internal metrics, the null-gate
  verdict. There is nothing to subtract from, so these are read as *behaviour
  along a sweep*: at what separation does the verdict flip, and does it also flip
  on a continuum, where flipping is wrong. A check with no counterpart can still
  be caught being confidently wrong; it just cannot be scored pointwise.

What is deliberately **not** here: Domain B. Order ablation, input faithfulness
and kNN concordance take a patient timeline and a model, not an embedding matrix.
A synthetic space has no input to ablate, so they are outside Tier 0's reach as
the generator stands and are calibrated, if at all, only on real data. A1
(vocabulary coverage) is out for the same reason -- there is no tokenizer.

Pure functions; the grid, the arms and the gate subset are declared in
``task_harness``.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from embedding_space_exploration.battery.check_embeddings import (
    embedding_checks,
    post_prep_anisotropy,
)
from embedding_space_exploration.battery.cluster import (
    choose_k,
    cluster_metrics,
    fit_clusters,
    prediction_strength_sweep,
)
from embedding_space_exploration.battery.cluster_tendency import (
    cluster_tendency_vs_null,
    null_gate_verdict,
    null_margin,
)
from embedding_space_exploration.battery.confound_decodability import (
    confound_decodability,
)
from embedding_space_exploration.battery.confound_orientation import (
    principal_component_regression,
)
from embedding_space_exploration.battery.prep import prepare_matrix
from embedding_space_exploration.battery.stability import stability_sweep
from embedding_space_exploration.config import COMPARISON_K, PRIMARY_SCALING
from embedding_space_exploration.data_management.splits import TRAIN, split_label

# The covariate the generator plants; every other covariate column is a decoy that
# loads on nothing, and the worst decoy is a check's false-positive floor.
CONFOUND = "log_n_events"


def score_space(
    space,
    *,
    scaling=PRIMARY_SCALING,
    run_null_gate=False,
    sweep_kwargs=None,
    stability_kwargs=None,
    gate_kwargs=None,
):
    """Run the battery on one synthetic space and score it against the truth.

    Args:
        space: The five frames the generator wrote -- ``embeddings``,
            ``covariates``, ``truth``, ``split``, ``spec``.
        scaling: Preprocessing arm (``prepare_matrix``). ``"raw"`` is the honest
            baseline, ``"spherical"`` the primary arm. Both are worth running:
            the arm is between the space and every clustering check, so a check's
            behaviour is a property of the pair, not of the space alone.
        run_null_gate: Run D1. It costs ``n_draws + 1`` full prediction-strength
            sweeps, which is why it is a declared subset rather than the default.
        sweep_kwargs: Overrides for ``prediction_strength_sweep``.
        stability_kwargs: Overrides for ``stability_sweep``.
        gate_kwargs: Overrides for ``cluster_tendency_vs_null``.

    The three override dicts all default to the frozen constants in ``config``,
    and **the task layer never sets them**. They exist so a test or an
    exploratory pass can run cheaply; a number that is going to be reported has
    to come from the defaults, because a gate calibrated cheaper than it will be
    used is not a calibration of that gate.

    Returns:
        ``{"summary": one-row frame, "curve": one row per k}``. The summary
        carries the headline of every check plus the truth comparisons; the curve
        carries the per-k prediction-strength, stability and null-band values
        that a sweep is plotted from.
    """
    embeddings, covariates = space["embeddings"], space["covariates"]
    truth, split, spec = space["truth"], space["split"], space["spec"].iloc[0]

    train = split_label(embeddings["person_id"], split) == TRAIN
    _, matrix = prepare_matrix(embeddings, fit_mask=train, scaling=scaling)

    sweep = prediction_strength_sweep(matrix[train], **(sweep_kwargs or {}))
    chosen_k = choose_k(sweep)
    stability = stability_sweep(matrix[train], **(stability_kwargs or {}))
    tendency = (
        cluster_tendency_vs_null(
            matrix[train],
            resphere_null=scaling == "spherical",
            **(gate_kwargs or {}),
        )
        if run_null_gate
        else None
    )

    summary = {
        "scaling": scaling,
        **_representation_row(embeddings, split, scaling),
        **_partition_row(matrix, train, sweep, stability, chosen_k),
        **_gate_row(tendency),
        **_confound_row(embeddings, covariates, split, train, scaling),
        **_truth_row(matrix, truth, spec, chosen_k),
    }
    summary |= _declared_vs_measured(summary, spec)
    return {
        "summary": pd.DataFrame([summary]),
        "curve": _curve(sweep, stability, tendency, scaling),
    }


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _representation_row(embeddings, split, scaling):
    """Domain A: the mechanical checks, which read the raw matrix, not the arm."""
    checks = embedding_checks(embeddings).iloc[0]
    row = {
        "n_nan_rows": int(checks["n_nan_rows"]),
        "n_constant_dims": int(checks["n_constant_dims"]),
        "n_duplicate_vectors": int(checks["n_duplicate_vectors"]),
        "mean_cosine_to_centroid": float(checks["mean_cosine_to_centroid"]),
        "rankme": float(checks["rankme"]),
        "rankme_ratio": float(checks["rankme_ratio"]),
        "mean_cosine_to_centroid_post_prep": np.nan,
    }
    # `post_prep_anisotropy` reports the primary arm by construction, so it is
    # only meaningful on the row that ran it.
    if scaling == PRIMARY_SCALING:
        post = post_prep_anisotropy(embeddings, split).iloc[0]
        row["mean_cosine_to_centroid_post_prep"] = float(
            post["mean_cosine_to_centroid_post_prep"]
        )
    return row


def _partition_row(matrix, train, sweep, stability, chosen_k):
    """Domains D2/D3/D4: what the partition looks like before truth is consulted."""
    labels = fit_clusters(matrix[train], chosen_k).predict(matrix)
    metrics = cluster_metrics(matrix[train], labels[train]).iloc[0]
    return {
        "chosen_k": int(chosen_k),
        "prediction_strength_at_chosen_k": _at_k(
            sweep, chosen_k, "prediction_strength_mean"
        ),
        "bootstrap_ari_at_chosen_k": _at_k(stability, chosen_k, "mean_ari"),
        "silhouette": float(metrics["silhouette"]),
        "calinski_harabasz": float(metrics["calinski_harabasz"]),
        "davies_bouldin": float(metrics["davies_bouldin"]),
    }


def _gate_row(tendency):
    """Domain D1, or a row of NaNs where the gate was not part of the subset."""
    if tendency is None:
        return {
            "gate_ran": False,
            "gate_verdict": None,
            "gate_any_k_beats_null": np.nan,
            "gate_max_headroom_margin": np.nan,
            "gate_k_at_max_margin": np.nan,
            "gate_margin_at_comparison_k": np.nan,
        }
    verdict = null_gate_verdict(tendency).iloc[0]
    scored = null_margin(tendency)
    return {
        "gate_ran": True,
        # The leading word is the verdict; the rest is the sentence explaining it.
        "gate_verdict": verdict["verdict"].split(":")[0],
        "gate_any_k_beats_null": bool(verdict["any_k_beats_null"]),
        # The continuous statistic the verdict is derived from. Recorded so the
        # threshold can move without re-running a single cell.
        "gate_max_headroom_margin": float(verdict["max_headroom_margin"]),
        # A diagnostic, not a selected k.
        "gate_k_at_max_margin": int(verdict["k_at_max_margin"]),
        # The margin at the k a cross-space comparison would actually be run at.
        "gate_margin_at_comparison_k": _at_k(scored, COMPARISON_K, "headroom_margin"),
    }


def _confound_row(embeddings, covariates, split, train, scaling):
    """C1 orientation beside probe decodability -- the dissociation, side by side."""
    per_covariate = principal_component_regression(
        embeddings,
        covariates,
        fit_mask=train,
        l2_normalize=scaling == "spherical",
    )["per_covariate"].set_index("covariate")
    decodability = confound_decodability(embeddings, covariates, split).set_index(
        "covariate"
    )
    decoys = [c for c in per_covariate.index if c != CONFOUND]

    return {
        "pcr_confound_variance_weighted_r2": float(
            per_covariate.loc[CONFOUND, "variance_weighted_r2"]
        ),
        "pcr_confound_leading_pc_r2": float(
            per_covariate.loc[CONFOUND, "leading_pc_r2"]
        ),
        "pcr_decoy_max_variance_weighted_r2": _worst_decoy(
            per_covariate, decoys, "variance_weighted_r2"
        ),
        "decode_confound_r2_linear": float(decodability.loc[CONFOUND, "r2_linear"]),
        "decode_confound_r2_nonlinear": float(
            decodability.loc[CONFOUND, "r2_nonlinear"]
        ),
        "decode_decoy_max_r2_nonlinear": _worst_decoy(
            decodability, decoys, "r2_nonlinear"
        ),
    }


def _truth_row(matrix, truth, spec, chosen_k):
    """The columns only a simulation can supply: reported next to planted."""
    planted = truth["cluster"]
    known = planted.notna().to_numpy()
    row = {
        "n_true_clusters": int(spec["n_clusters"]),
        "k_error": np.nan,
        "ari_at_chosen_k": np.nan,
        "ari_at_true_k": np.nan,
    }
    if not known.any():
        # A continuum has no planted partition. That is not a gap in the scoring:
        # the correct answer is "no clusters", so the gate verdict carries it and
        # an ARI against nothing would be a category error.
        return row

    labels = np.asarray(planted[known], dtype="int64")
    row["k_error"] = int(chosen_k) - int(spec["n_clusters"])
    row["ari_at_chosen_k"] = _ari(matrix[known], labels, chosen_k)
    # Separates "found the structure" from "found the right k": a space can score
    # a poor ARI purely because k was chosen badly.
    row["ari_at_true_k"] = _ari(matrix[known], labels, int(spec["n_clusters"]))
    return row


def _declared_vs_measured(summary, spec):
    """Measured minus declared, for the checks that have a planted counterpart.

    Kept apart from ``_truth_row`` because these two hold for *every* cell:
    anisotropy and intrinsic dimension are planted whether or not there is a
    partition, so a continuum has an anisotropy error like anything else.
    """
    return {
        "anisotropy_error": round(
            summary["mean_cosine_to_centroid"] - float(spec["anisotropy"]), 4
        ),
        "rankme_minus_intrinsic_dim": round(
            summary["rankme"] - float(spec["intrinsic_dim"]), 4
        ),
    }


def _ari(matrix, planted, k):
    """ARI between the planted labels and a k-means partition at the given k."""
    return round(
        float(adjusted_rand_score(planted, fit_clusters(matrix, k).labels_)), 4
    )


def _curve(sweep, stability, tendency, scaling):
    """Per-k values from every sweep-shaped check, joined on k."""
    curve = sweep.merge(stability, on="k", how="outer")
    if tendency is not None:
        curve = curve.merge(tendency, on="k", how="outer")
    curve.insert(0, "scaling", scaling)
    return curve


def _at_k(frame, k, column):
    """One column of a per-k frame at a given k, or NaN if that k is not swept."""
    row = frame[frame["k"] == k]
    return float(row[column].iloc[0]) if not row.empty else np.nan


def _worst_decoy(frame, decoys, column):
    """The highest value any pure-noise covariate reached: the false-positive floor."""
    present = [c for c in decoys if c in frame.index]
    return float(frame.loc[present, column].max()) if present else np.nan
