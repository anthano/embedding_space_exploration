import numpy as np
import pandas as pd
import pytest

from embedding_space_exploration.battery.confound_orientation import (
    principal_component_regression,
)


def _embeddings(matrix):
    cols = {f"dim_{i}": matrix[:, i] for i in range(matrix.shape[1])}
    return pd.DataFrame({"person_id": range(len(matrix)), **cols})


def test_continuous_confound_lands_on_the_leading_pc():
    # dim_0 is a loud copy of `confound`; every other dim is quiet noise. So PC0 is
    # essentially the confound axis, and PCR must attribute it: dominant covariate at
    # PC0 is `confound`, and it carries far more embedding variance than `noise`.
    rng = np.random.default_rng(0)
    n = 300
    confound = rng.normal(size=n)
    matrix = rng.normal(size=(n, 6)) * 0.1
    matrix[:, 0] += 6.0 * confound
    covariates = pd.DataFrame(
        {"person_id": range(n), "confound": confound, "noise": rng.normal(size=n)}
    )

    out = principal_component_regression(_embeddings(matrix), covariates)
    per_cov = out["per_covariate"].set_index("covariate")
    per_pc = out["per_pc"].set_index("pc")

    assert per_pc.loc[0, "dominant_covariate"] == "confound"
    assert per_cov.loc["confound", "top_pc"] == pytest.approx(0)
    assert (
        per_cov.loc["confound", "variance_weighted_r2"]
        > per_cov.loc["noise", "variance_weighted_r2"]
    )


def test_categorical_confound_uses_eta_squared():
    # A 3-level category that shifts dim_0 by group -> eta^2 on PC0 is high.
    rng = np.random.default_rng(1)
    n = 300
    group = rng.integers(0, 3, size=n)
    matrix = rng.normal(size=(n, 5)) * 0.1
    matrix[:, 0] += 5.0 * group
    covariates = pd.DataFrame(
        {"person_id": range(n), "grp": pd.Categorical(group.astype(str))}
    )

    out = principal_component_regression(_embeddings(matrix), covariates)
    grp_leading_r2 = (
        out["per_covariate"].set_index("covariate").loc["grp", "leading_pc_r2"]
    )
    assert out["per_pc"].set_index("pc").loc[0, "dominant_covariate"] == "grp"
    assert grp_leading_r2 == pytest.approx(1.0, abs=0.4)  # eta^2 ~ 1 on the group axis


def test_l2_normalize_path_runs_and_returns_both_frames():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(80, 5))
    covariates = pd.DataFrame({"person_id": range(80), "x": rng.normal(size=80)})
    out = principal_component_regression(
        _embeddings(matrix), covariates, l2_normalize=True
    )
    assert set(out) == {"per_pc", "per_covariate"}
    assert "variance_weighted_r2" in out["per_covariate"].columns
    assert not out["per_pc"].empty
