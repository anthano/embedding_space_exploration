import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from embedding_space_exploration.analysis.probe import (
    bootstrap_auroc_ci,
    fit_probe,
)
from embedding_space_exploration.data_management.splits import TEST, TRAIN, VAL


def _separable(n_per_split=120, n_dims=8, seed=0):
    """A signal one dimension carries, split evenly into train / val / test."""
    rng = np.random.default_rng(seed)
    n = n_per_split * 3
    y = np.tile([0, 1], n // 2)
    features = rng.normal(size=(n, n_dims))
    features[:, 0] += 2.0 * y
    split = np.array([TRAIN, VAL, TEST] * (n // 3))
    return features, y, split


def test_a_learnable_signal_scores_above_chance():
    features, y, split = _separable()
    result = fit_probe(features, y, split, n_bootstrap=50)
    assert result["auroc"] > 0.8


def test_pure_noise_lands_near_chance():
    rng = np.random.default_rng(1)
    n = 360
    features = rng.normal(size=(n, 8))
    y = np.tile([0, 1], n // 2)
    split = np.array([TRAIN, VAL, TEST] * (n // 3))
    result = fit_probe(features, y, split, n_bootstrap=50)
    assert result["auroc"] == pytest.approx(0.5, abs=0.15)


def test_c_is_chosen_on_val_and_never_on_test():
    # The protocol's whole point: `val` selects, `test` is scored once. If C were
    # tuned on test, restricting the grid to a single value could not change the
    # selected C while leaving the test score free to improve.
    features, y, split = _separable()
    tuned = fit_probe(features, y, split, n_bootstrap=0)
    fixed = fit_probe(features, y, split, c_grid=(tuned["c"],), n_bootstrap=0)
    assert fixed["c"] == tuned["c"]
    assert fixed["auroc"] == pytest.approx(tuned["auroc"])
    assert not np.isnan(tuned["val_auroc"])


def test_the_split_sizes_are_reported_for_every_role():
    features, y, split = _separable(n_per_split=90)
    result = fit_probe(features, y, split, n_bootstrap=0)
    assert (result["n_train"], result["n_val"], result["n_test"]) == (90, 90, 90)


def test_a_single_class_split_returns_nan_rather_than_raising():
    # A small slice can leave a split single-class. That is not a failure and
    # must not take the whole run down with it.
    features = np.random.default_rng(2).normal(size=(60, 4))
    y = np.zeros(60, dtype=int)
    split = np.array([TRAIN, VAL, TEST] * 20)
    result = fit_probe(features, y, split)
    assert np.isnan(result["auroc"])
    assert result["n_train"] == 20


def test_the_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    y = np.tile([0, 1], 150)
    scores = rng.uniform(size=300) + 0.35 * y
    low, high = bootstrap_auroc_ci(y, scores, n_bootstrap=200)
    assert low < roc_auc_score(y, scores) < high


def test_the_interval_is_reproducible_under_a_fixed_seed():
    y = np.tile([0, 1], 100)
    scores = np.random.default_rng(4).uniform(size=200) + 0.3 * y
    first = bootstrap_auroc_ci(y, scores, n_bootstrap=100, random_state=7)
    second = bootstrap_auroc_ci(y, scores, n_bootstrap=100, random_state=7)
    assert first == second


def test_single_class_resamples_are_dropped_not_scored_as_chance():
    # One positive row means most resamples miss it entirely. Those have no
    # AUROC; counting them as 0.5 would drag the interval toward chance by an
    # amount that depends on nothing but class balance.
    y = np.array([0] * 40 + [1])
    scores = np.linspace(0, 1, 41)
    low, high = bootstrap_auroc_ci(y, scores, n_bootstrap=200)
    assert low > 0.5
    assert high <= 1.0
