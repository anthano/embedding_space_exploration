"""Assign cohort patients to a derivation / development / lockbox split.

Pure functions only; reading the cohort and writing ``split.parquet`` belong in a
``task_split_cohort.py``. The split is the single source of truth threaded through
every downstream step that *fits* or *selects* something on the cohort, so held-out
patients never influence derivation **or** model/variant selection.

Three roles:

- ``derivation``  (~60%): fit everything cohort-dependent -- rare-code vocab, PCA
  basis, K-means centroids, any whitening/contrastive parameters.
- ``development`` (~20%): the only held-out set used to *choose between* pipeline
  variants (anisotropy correction, k, model, pooling) and to inspect cross-tabs.
  "Used up" by selection, so it cannot support the final claim.
- ``lockbox``     (~20%): touched exactly once, at the end, for the confirmatory
  analysis. Never inspected during development.

Note the frozen-model embedding step needs no split -- each patient is embedded in
isolation, so nothing is learned across patients.

Ported from the All of Us project, where the lockbox backs a single confirmatory
outcome test. **Whether this three-way split is the right shape for the EHRSHOT
tier is an open design decision** (EHRSHOT publishes its own train/val/test
alongside baseline AUROCs). The module is here and tested; nothing depends on it
yet, and the dataset-specific bindings below are parameters rather than defaults.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Seeded 60/20/20. Random rather than geographic: a random split is the field
# precedent and it avoids confounding subtype transport with demographic shift. The
# held-out 40% is halved into development / lockbox.
SPLIT_SEED = 0
HELD_OUT_SIZE = 0.4  # development + lockbox combined; the rest is derivation
LOCKBOX_FRACTION_OF_HELD_OUT = 0.5  # split the held-out half into dev / lockbox

DERIVATION = "derivation"
DEVELOPMENT = "development"
LOCKBOX = "lockbox"

# The splits any *non-confirmatory* step may read. Everything that fits, selects,
# inspects or cross-tabs works from these two; the lockbox is opened exactly once.
OPEN_SPLITS = (DERIVATION, DEVELOPMENT)

# The seal is mechanical, not a promise. `visible_splits` / `drop_lockbox` are the
# single choke point every confirmatory step goes through, and they exclude the
# lockbox unless this env var is set to "1" for that one run. Making the unsealing an
# explicit, greppable act -- rather than trusting each task to remember a filter -- is
# the point: the pre-registration says "touched exactly once", and a filter you have
# to opt out of is the only version of that claim you can audit.
UNSEAL_LOCKBOX_ENV = "EMBEDDING_SPACE_UNSEAL_LOCKBOX"

# Age bands used only as the fallback stratification key, when the caller declares
# no dataset-specific one.
_AGE_BINS = (0, 18, 30, 45, 60, 120)
MIN_STRATUM_COUNT = 2
_MIN_STRATA = 2  # need >= 2 non-empty strata for stratification to mean anything


def assign_split(
    cohort,
    *,
    held_out_size=HELD_OUT_SIZE,
    lockbox_fraction=LOCKBOX_FRACTION_OF_HELD_OUT,
    seed=SPLIT_SEED,
    stratify_cols=(),
    id_col="person_id",
):
    """Assign each cohort patient to ``derivation`` / ``development`` / ``lockbox``.

    A seeded, stratified two-stage split: hold out ``held_out_size`` of patients
    (stratified), then divide that hold-out into development and lockbox by
    ``lockbox_fraction`` (stratified again). Stratifies on the first available
    column in ``stratify_cols``; if none is given or present, falls back to an age
    band from ``age_at_index``, and to an unstratified split if even that is
    unavailable or too sparse.

    ``stratify_cols`` defaults to empty on purpose. Which label the split should be
    balanced on is dataset-specific and, for the primary tier, still an open design
    decision -- so each dataset's task declares its own key rather than silently
    inheriting another cohort's.

    Args:
        cohort: Cohort frame with at least ``id_col`` (and ideally a stratification
            column / ``age_at_index``).
        held_out_size: Combined development + lockbox fraction; the remainder is
            derivation.
        lockbox_fraction: Fraction of the held-out set assigned to the lockbox (the
            rest becomes development).
        seed: RNG seed for a reproducible split. Re-deriving after the cohort grows
            reshuffles membership; re-run downstream steps if so.
        stratify_cols: Preferred stratification columns, in priority order.
        id_col: Patient-id column in ``cohort``.

    Returns:
        DataFrame with ``id_col`` and ``split`` (one of ``derivation`` /
        ``development`` / ``lockbox``), one row per patient, in the cohort's order.
    """
    # Split on positional indices so the second-stage strata stay aligned to the
    # held-out rows (splitting ids while deriving strata from a re-filtered frame
    # would misalign labels to ids).
    strata = _strata(cohort, stratify_cols)
    strata = None if strata is None else np.asarray(strata)
    idx = np.arange(len(cohort))

    deriv_idx, held_idx = train_test_split(
        idx, test_size=held_out_size, random_state=seed, stratify=strata
    )
    held_strata = None if strata is None else strata[held_idx]
    dev_idx, lock_idx = train_test_split(
        held_idx, test_size=lockbox_fraction, random_state=seed, stratify=held_strata
    )

    label = np.empty(len(cohort), dtype=object)
    label[deriv_idx] = DERIVATION
    label[dev_idx] = DEVELOPMENT
    label[lock_idx] = LOCKBOX
    return pd.DataFrame({id_col: cohort[id_col].to_numpy(), "split": label})


def lockbox_is_sealed():
    """Whether the lockbox is currently sealed (the default, always, except once)."""
    return os.getenv(UNSEAL_LOCKBOX_ENV, "0") != "1"


def visible_splits():
    """Split labels the current run is allowed to read.

    ``OPEN_SPLITS`` normally; all three only when
    ``EMBEDDING_SPACE_UNSEAL_LOCKBOX=1`` marks the single pre-registered
    confirmatory run.
    """
    if lockbox_is_sealed():
        return OPEN_SPLITS
    return (*OPEN_SPLITS, LOCKBOX)


def drop_lockbox(frame, split, *, on="person_id"):
    """Restrict ``frame`` to the splits this run may read, adding a ``split`` column.

    Args:
        frame: Any per-patient frame carrying ``on``.
        split: The ``split.parquet`` frame (patient id, ``split``).
        on: Patient-id column, in both ``frame`` and ``split``.

    Returns:
        ``frame`` filtered to ``visible_splits()``, with ``split`` attached. Patients
        absent from ``split`` are dropped -- an unlabelled patient cannot be shown to
        be outside the lockbox, so the safe reading is that it might be inside.
    """
    labelled = frame.copy()
    labelled["split"] = labelled[on].map(split.set_index(on)["split"])
    return labelled[labelled["split"].isin(visible_splits())].reset_index(drop=True)


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _strata(cohort, stratify_cols):
    """Pick a stratification key, or None if no usable one exists.

    Returns the first present ``stratify_cols`` column, else an age band from
    ``age_at_index``. Falls back to ``None`` (unstratified) if the candidate has
    missing values or any class with fewer than two members -- stratified splitting
    needs >= 2 per class, so a rare stratum must not break the build.
    """
    candidate = next((cohort[c] for c in stratify_cols if c in cohort.columns), None)
    if candidate is None and "age_at_index" in cohort.columns:
        candidate = pd.cut(cohort["age_at_index"], bins=_AGE_BINS)
    if candidate is None or candidate.isna().any():
        return None
    # `pd.cut` yields a categorical whose value_counts includes 0 for unused bins;
    # count only *observed* strata so empty age bands don't disable stratification.
    counts = candidate.value_counts()
    counts = counts[counts > 0]
    if len(counts) < _MIN_STRATA or counts.min() < MIN_STRATUM_COUNT:
        return None
    return candidate
