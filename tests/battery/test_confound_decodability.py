import numpy as np
import pandas as pd

from embedding_space_exploration.battery.confound_decodability import (
    confound_decodability,
)
from embedding_space_exploration.simulation.generator import simulate_embeddings


def _space(**kwargs):
    return simulate_embeddings(
        n_patients=800,
        n_dims=16,
        intrinsic_dim=3,
        n_clusters=1,
        separation=0.0,
        noise=0.05,
        n_decoy_covariates=1,
        **kwargs,
    )


def test_a_radial_confound_is_invisible_to_the_linear_probe_and_not_the_other():
    # The whole reason this module exists beside C1: no linear direction carries
    # the confound, so both PCR and a linear probe read ~0 while the nonlinear
    # probe recovers it. Reporting only the linear reading would call this space
    # clean.
    space = _space(confound_orientation="radial", confound_strength=4.0)
    report = confound_decodability(
        space["embeddings"], space["covariates"], space["split"]
    ).set_index("covariate")

    assert report.loc["log_n_events", "r2_linear"] < 0.1
    assert report.loc["log_n_events", "r2_nonlinear"] > 0.5
    assert report.loc["log_n_events", "nonlinear_gain"] > 0.4


def test_an_axis_confound_is_decodable_by_both():
    space = _space(confound_orientation="axis", confound_strength=4.0)
    report = confound_decodability(
        space["embeddings"], space["covariates"], space["split"]
    ).set_index("covariate")

    assert report.loc["log_n_events", "r2_linear"] > 0.9
    assert report.loc["log_n_events", "r2_nonlinear"] > 0.8


def test_a_decoy_that_loads_on_nothing_sets_the_floor():
    space = _space(confound_orientation="axis", confound_strength=4.0)
    report = confound_decodability(
        space["embeddings"], space["covariates"], space["split"]
    ).set_index("covariate")

    # Held-out R^2 on pure noise sits at or below zero -- anything a real
    # covariate scores has to clear this to mean anything.
    assert report.loc["decoy_0", "r2_nonlinear"] < 0.1
    assert report.loc["decoy_0", "r2_linear"] < 0.1


def test_non_numeric_covariates_are_skipped_not_guessed_at():
    space = _space(confound_orientation="axis", confound_strength=2.0)
    covariates = space["covariates"].assign(
        site=np.where(np.arange(len(space["covariates"])) % 2, "a", "b")
    )
    report = confound_decodability(space["embeddings"], covariates, space["split"])
    assert "site" not in set(report["covariate"])


def test_the_probe_is_fit_and_scored_on_disjoint_splits():
    space = _space(confound_orientation="axis", confound_strength=2.0)
    report = confound_decodability(
        space["embeddings"], space["covariates"], space["split"]
    )
    counts = space["split"]["split"].value_counts()
    assert (report["n_fit"] == counts["train"]).all()
    assert (report["n_score"] == counts["val"]).all()
    assert pd.notna(report["r2_nonlinear"]).all()
