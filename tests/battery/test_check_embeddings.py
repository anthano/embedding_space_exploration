import numpy as np
import pandas as pd
import pytest

from embedding_space_exploration.battery.check_embeddings import embedding_checks


def _frame(matrix):
    cols = {f"dim_{i}": matrix[:, i] for i in range(matrix.shape[1])}
    return pd.DataFrame({"person_id": range(len(matrix)), **cols})


def test_rankme_collapses_to_one_for_rank_one_space():
    # A rank-1 matrix (every row a multiple of one direction) is the extreme cone:
    # all variance on a single axis, so RankMe must be ~1.
    direction = np.array([1.0, 2.0, -1.0, 0.5])
    matrix = np.outer(np.arange(1, 21, dtype="float64"), direction)
    rankme = embedding_checks(_frame(matrix))["rankme"].iloc[0]
    assert rankme == pytest.approx(1.0, abs=0.1)


def test_rankme_is_higher_for_an_isotropic_space():
    # An isotropic block (identity-covariance Gaussian) uses many dimensions, so its
    # effective rank far exceeds the rank-1 case and its ratio is well above zero.
    rng = np.random.default_rng(0)
    n_dims = 16
    matrix = rng.standard_normal((500, n_dims))
    row = embedding_checks(_frame(matrix)).iloc[0]
    assert row["rankme"] > n_dims / 2  # uses well over half the dimensions
    assert 0.0 < row["rankme_ratio"] <= 1.0
