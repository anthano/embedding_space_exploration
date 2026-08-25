import numpy as np
import pandas as pd

from embedding_space_exploration.splits import (
    OPEN_SPLITS,
    TEST,
    TRAIN,
    VAL,
    fit_mask,
    split_label,
)


def _split_frame():
    return pd.DataFrame(
        {"person_id": [10, 11, 12, 13], "split": [TRAIN, VAL, TEST, TRAIN]}
    )


def test_labels_follow_row_order_not_split_frame_order():
    # The embedding frame's order is authoritative; the split frame is a lookup.
    person_id = pd.Series([13, 10, 12, 11])

    labels = split_label(person_id, _split_frame())

    assert list(labels) == [TRAIN, TRAIN, TEST, VAL]


def test_only_train_rows_are_fit_on():
    mask = fit_mask(pd.Series([10, 11, 12, 13]), _split_frame())

    assert list(mask) == [True, False, False, True]


def test_unknown_person_matches_no_role():
    # A patient absent from the split map must never silently become a fit row.
    mask = fit_mask(pd.Series([10, 999]), _split_frame())

    assert list(mask) == [True, False]
    assert np.isnan(split_label(pd.Series([999]), _split_frame())[0])


def test_test_split_is_not_open():
    assert set(OPEN_SPLITS) == {TRAIN, VAL}
    assert TEST not in OPEN_SPLITS
