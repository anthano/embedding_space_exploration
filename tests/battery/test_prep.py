import numpy as np
import pandas as pd
from numpy.linalg import norm

from embedding_space_exploration.battery.prep import prepare_matrix


def _embeddings(n=60, d=10, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(size=(n, d)), columns=[f"dim_{i}" for i in range(d)]
    )
    frame.insert(0, "person_id", range(n))
    return frame


def test_prepare_matrix_rows_are_l2_normalized():
    _, matrix = prepare_matrix(_embeddings(), n_components=5)
    assert np.allclose(norm(matrix, axis=1), 1.0)


def test_raw_scaling_is_the_untouched_baseline():
    # The raw baseline returns the model output verbatim -- no L2, no PCA, magnitude
    # retained -- so the arms (spherical, ...) have something honest to beat.
    emb = _embeddings(n=40, d=8)
    person_id, matrix = prepare_matrix(emb, scaling="raw")
    dims = [c for c in emb.columns if c.startswith("dim_")]
    assert np.array_equal(matrix, emb[dims].to_numpy(dtype="float64"))
    assert np.array_equal(person_id, emb["person_id"].to_numpy())


def test_fit_mask_changes_the_projection():
    emb = _embeddings()
    mask = np.zeros(len(emb), dtype=bool)
    mask[:30] = True  # fit the PCA basis on the first half (the "train" rows)

    _, all_fit = prepare_matrix(emb, n_components=5)
    _, train_fit = prepare_matrix(emb, fit_mask=mask, n_components=5)

    assert all_fit.shape == train_fit.shape == (60, 5)
    # a train-only basis is not the same as one fit on every patient
    assert not np.allclose(all_fit, train_fit)
