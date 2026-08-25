"""The train / val / test contract every dataset in the project is read through.

One vocabulary across all three datasets, so a reader learns the convention once
and it holds for EHRSHOT (Tier 1), All of Us (Tier 3) and MIMIC-IV (Tier 3b).
Three roles:

- ``train``: fit everything cohort-dependent -- PCA basis, K-means centroids, probe
  weights, any whitening or contrastive parameters.
- ``val``: choose *between* pipeline variants (preprocessing arm, k, model, pooling)
  and inspect cross-tabs. Used up by selection, so it cannot support a final claim.
- ``test``: the reported numbers.

The names are EHRSHOT's, taken from its shipped ``person_id_map.csv``
(``omop_person_id -> train | val | test``). Only the *vocabulary* transfers across
datasets -- the assignments are patient-level and the cohorts are disjoint, so All of
Us and MIMIC-IV each derive their own three-way split under these role names.

Note this supersedes the ``derivation`` / ``development`` / ``lockbox`` split of the
All of Us work, whose ``ALLOFUS_UNSEAL_LOCKBOX`` seal was a *mechanical* guard against
selection leakage. Nothing here reproduces that gate: with it retired, the frozen
constants in ``config`` and the selection-budget ledger carry that burden alone.

Note also that the frozen-model embedding step needs no split -- each patient is
embedded in isolation, so nothing is learned across patients.
"""

import numpy as np

TRAIN = "train"
VAL = "val"
TEST = "test"

# The splits any *non-confirmatory* step may read. Everything that fits, selects,
# inspects or cross-tabs works from these two.
OPEN_SPLITS = (TRAIN, VAL)


def split_label(person_id, split):
    """Map each row's ``person_id`` to its split label.

    Args:
        person_id: Series of person ids, in the row order of the frame being
            labelled.
        split: Frame with ``person_id`` and ``split`` columns.

    Returns:
        Array of split labels aligned with ``person_id``. Rows whose id is absent
        from ``split`` come back as NaN and so match no role.
    """
    return person_id.map(split.set_index("person_id")["split"]).to_numpy()


def fit_mask(person_id, split):
    """Boolean mask selecting the rows anything cohort-dependent may be fit on.

    The single choke point for "fit on train only", so held-out patients never
    inform a PCA basis, a scaler or a set of centroids.

    Args:
        person_id: Series of person ids, in the row order of the frame being fit.
        split: Frame with ``person_id`` and ``split`` columns.

    Returns:
        Boolean array, ``True`` for the ``train`` rows.
    """
    return np.asarray(split_label(person_id, split) == TRAIN)
