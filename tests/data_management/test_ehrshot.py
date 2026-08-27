import numpy as np
import pandas as pd

from embedding_space_exploration.data_management.ehrshot import align_features


def _index(pairs):
    return pd.DataFrame(
        {
            "person_id": [p for p, _ in pairs],
            "prediction_time": pd.to_datetime([t for _, t in pairs]),
        }
    )


def test_features_come_back_in_the_labels_row_order_not_the_index_order():
    # The join must not silently reorder: `fit_probe` pairs these rows with the
    # label vector positionally, so a reordering here mislabels every row.
    index = _index([(1, "2020-01-01"), (2, "2020-01-01"), (3, "2020-01-01")])
    matrix = np.array([[10.0], [20.0], [30.0]])
    labels = _index([(3, "2020-01-01"), (1, "2020-01-01")])

    features, matched = align_features(index, matrix, labels)

    assert matched.all()
    assert features.ravel().tolist() == [30.0, 10.0]


def test_a_patient_is_matched_on_time_as_well_as_id():
    # The lab tasks label the same patient at hundreds of prediction times, so
    # the id alone does not identify a vector.
    index = _index([(1, "2020-01-01"), (1, "2021-06-30")])
    matrix = np.array([[1.0], [2.0]])
    labels = _index([(1, "2021-06-30")])

    features, _ = align_features(index, matrix, labels)

    assert features.ravel().tolist() == [2.0]


def test_unmatched_rows_are_flagged_rather_than_dropped_silently():
    # A few labelled patients are absent from the release's feature matrix. The
    # caller has to be told which, not handed a shorter frame to discover later.
    index = _index([(1, "2020-01-01")])
    matrix = np.array([[1.0]])
    labels = _index([(1, "2020-01-01"), (99, "2020-01-01")])

    features, matched = align_features(index, matrix, labels)

    assert matched.tolist() == [True, False]
    assert len(features) == 1


def test_nothing_matching_gives_an_empty_result_not_an_error():
    index = _index([(1, "2020-01-01")])
    matrix = np.array([[1.0]])
    labels = _index([(2, "2020-01-01")])

    features, matched = align_features(index, matrix, labels)

    assert not matched.any()
    assert len(features) == 0
