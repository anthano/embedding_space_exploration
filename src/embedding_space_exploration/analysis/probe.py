"""Domain U, Y1: the supervised linear probe on a frozen representation.

Built to the protocol [[Study Design Freeze]] section 6 pins verbatim from
[[Wornow2025]], because the published-baseline oracle is only valid if our
protocol is the one that produced those numbers. The three split roles are
theirs: ``train`` fits the head, ``val`` tunes the hyperparameter, ``test`` is
scored once.

Deliberately generic. It takes a feature matrix and a label vector and knows
nothing about EHRSHOT, so the same probe scores the shipped reference features,
our own extracted embeddings, and the Tier 1.3 baselines -- the comparison is
only meaningful if the head is identical across them.

The ``"All"`` data setting is what the headline numbers and the oracle use. The
few-shot arm (k in {8, 16, 32, 64, 128}) is secondary in section 6 and is not
built here yet.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from embedding_space_exploration.config import RANDOM_STATE
from embedding_space_exploration.data_management.splits import TEST, TRAIN, VAL

# Section 9: every headline metric carries a bootstrapped 95% CI over 1,000
# resamples of the test set, raised from 200 to match [[Wornow2025]] so our
# intervals sit alongside the published ones rather than merely near them.
N_BOOTSTRAP_METRIC = 1000

# Regularisation grid searched on the `val` split. Section 6 pins *that* C is
# tuned on val but not which values are searched, so this grid is our own
# declared choice rather than something read from the paper -- recorded here so
# it is a constant in the ledger's sense and not an argument someone varies.
PROBE_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

# Minimum distinct classes for an AUROC to exist at all.
_BINARY = 2


def fit_probe(
    features,
    y,
    split,
    *,
    c_grid=PROBE_C_GRID,
    n_bootstrap=N_BOOTSTRAP_METRIC,
    random_state=RANDOM_STATE,
):
    """Fit the probe on ``train``, tune C on ``val``, score AUROC on ``test``.

    Args:
        features: ``(n_rows, n_dims)`` frozen representation.
        y: Binary labels, aligned to ``features``.
        split: Split label per row -- ``train`` / ``val`` / ``test``.
        c_grid: Inverse regularisation strengths searched on ``val``.
        n_bootstrap: Test-set resamples for the CI. Zero skips the CI.
        random_state: Seed for the bootstrap and the solver.

    Returns:
        Dict with ``auroc``, ``ci_low``, ``ci_high``, ``c``, ``val_auroc`` and
        the three split sizes. ``auroc`` is NaN when a split is single-class,
        which a small slice can produce and which is not a failure.
    """
    # float32 throughout: the largest task is 318k x 768, and float64 would be
    # 2 GB before sklearn's own copies. The head is regularised logistic
    # regression, so the precision is immaterial to the AUROC.
    features = np.asarray(features, dtype=np.float32)
    y = np.asarray(y)
    split = np.asarray(split)
    train, val, test = (split == TRAIN), (split == VAL), (split == TEST)

    result = {
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "c": np.nan,
        "val_auroc": np.nan,
        "auroc": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
    }
    if not _scorable(y, train) or not _scorable(y, test):
        return result

    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train])

    best_c, best_val = _tune(
        x_train, y[train], scaler.transform(features[val]), y[val], c_grid, random_state
    )
    result["c"], result["val_auroc"] = best_c, best_val

    model = _fit(x_train, y[train], best_c, random_state)
    scores = model.predict_proba(scaler.transform(features[test]))[:, 1]
    result["auroc"] = float(roc_auc_score(y[test], scores))

    if n_bootstrap:
        low, high = bootstrap_auroc_ci(
            y[test], scores, n_bootstrap=n_bootstrap, random_state=random_state
        )
        result["ci_low"], result["ci_high"] = low, high
    return result


def bootstrap_auroc_ci(
    y, scores, *, n_bootstrap=N_BOOTSTRAP_METRIC, alpha=0.05, random_state=RANDOM_STATE
):
    """Percentile bootstrap CI for AUROC, resampling the scored rows.

    Resamples are drawn with replacement over the test rows, which is what
    [[Wornow2025]] does. A resample that lands single-class has no AUROC and is
    dropped rather than counted as 0.5, since counting it would pull every
    interval toward chance by an amount that depends on class balance.

    Args:
        y: Binary labels for the scored rows.
        scores: Predicted probabilities for the positive class.
        n_bootstrap: Number of resamples.
        alpha: Two-sided level; 0.05 gives a 95% interval.
        random_state: Seed.

    Returns:
        ``(low, high)``, or ``(nan, nan)`` if no resample was scorable.
    """
    y = np.asarray(y)
    scores = np.asarray(scores)
    rng = np.random.default_rng(random_state)
    draws = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < _BINARY:
            continue
        draws.append(roc_auc_score(y[idx], scores[idx]))
    if not draws:
        return float("nan"), float("nan")
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _scorable(y, mask):
    """Whether a split holds both classes, so an AUROC exists on it."""
    return bool(mask.any()) and len(np.unique(y[mask])) >= _BINARY


def _fit(x, y, c, random_state):
    """The head itself. One definition, so tuning and scoring cannot diverge."""
    return LogisticRegression(C=c, max_iter=1000, random_state=random_state).fit(x, y)


def _tune(x_train, y_train, x_val, y_val, c_grid, random_state):
    """Pick C by AUROC on ``val``.

    Falls back to the middle of the grid when ``val`` cannot be scored -- a
    single-class validation split is possible on a small slice, and refusing to
    return a model there would make the slice untestable.
    """
    if len(np.unique(y_val)) < _BINARY:
        return c_grid[len(c_grid) // 2], float("nan")
    scored = [
        (
            roc_auc_score(
                y_val,
                _fit(x_train, y_train, c, random_state).predict_proba(x_val)[:, 1],
            ),
            c,
        )
        for c in c_grid
    ]
    best_auroc, best_c = max(scored)
    return best_c, float(best_auroc)
