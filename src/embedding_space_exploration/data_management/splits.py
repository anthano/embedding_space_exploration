import numpy as np

TRAIN = "train"
VAL = "val"
TEST = "test"
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

    Args:
        person_id: Series of person ids, in the row order of the frame being fit.
        split: Frame with ``person_id`` and ``split`` columns.

    Returns:
        Boolean array, ``True`` for the ``train`` rows.
    """
    return np.asarray(split_label(person_id, split) == TRAIN)
