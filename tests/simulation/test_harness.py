import numpy as np
import pytest

from embedding_space_exploration.simulation.generator import simulate_embeddings
from embedding_space_exploration.simulation.grid import (
    GATE_BOTH_ARMS,
    PRIMARY_ARM,
    runs_null_gate,
)
from embedding_space_exploration.simulation.harness import score_space

# Cheap sweep settings. Never used by the task layer -- reported numbers come
# from the frozen constants.
_FAST = {
    "sweep_kwargs": {"k_values": (2, 3, 4, 5), "n_repeats": 3},
    "stability_kwargs": {"k_values": (2, 3, 4, 5), "n_boot": 3},
    "gate_kwargs": {"n_draws": 3, "k_values": (2, 3, 4, 5), "n_repeats": 3},
}


def _space(**kwargs):
    return simulate_embeddings(
        n_patients=600, n_dims=24, intrinsic_dim=8, n_clusters=3, **kwargs
    )


def test_a_clean_space_is_scored_as_recovered():
    summary = score_space(_space(separation=8.0), **_FAST)["summary"].iloc[0]
    assert summary["ari_at_true_k"] > 0.95
    assert summary["k_error"] == 0
    assert summary["n_nan_rows"] == 0


def test_the_declared_knobs_come_back_as_errors_not_raw_numbers():
    # What makes this a calibration rather than a measurement: the row carries
    # measured-minus-planted, so a sweep is an error curve.
    summary = score_space(_space(separation=6.0, anisotropy=0.6), **_FAST)[
        "summary"
    ].iloc[0]
    assert summary["anisotropy_error"] == pytest.approx(0.0, abs=0.06)
    assert np.isfinite(summary["rankme_minus_intrinsic_dim"])


def test_the_confound_dissociation_survives_the_round_trip():
    # The harness has to keep C1 and the probe apart, or the headline Tier 0
    # reading collapses into one number.
    radial = score_space(
        _space(
            separation=0.0,
            confound_orientation="radial",
            confound_strength=4.0,
            noise=0.05,
        ),
        **_FAST,
    )["summary"].iloc[0]

    assert radial["pcr_confound_variance_weighted_r2"] < 0.05
    assert radial["decode_confound_r2_linear"] < 0.1
    assert radial["decode_confound_r2_nonlinear"] > 0.5


def test_a_continuum_is_scored_without_inventing_a_partition():
    summary = score_space(_space(structure="continuum", separation=6.0), **_FAST)[
        "summary"
    ].iloc[0]

    # No planted partition, so ARI is a category error and stays absent...
    assert np.isnan(summary["ari_at_chosen_k"])
    assert np.isnan(summary["ari_at_true_k"])
    # ...but anisotropy and intrinsic dimension are planted regardless.
    assert np.isfinite(summary["anisotropy_error"])
    assert np.isfinite(summary["rankme_minus_intrinsic_dim"])


def test_the_gate_is_off_unless_asked_for():
    scored = score_space(_space(separation=6.0), **_FAST)["summary"].iloc[0]
    assert not scored["gate_ran"]
    assert scored["gate_verdict"] is None

    gated = score_space(_space(separation=6.0), run_null_gate=True, **_FAST)[
        "summary"
    ].iloc[0]
    assert gated["gate_ran"]
    assert gated["gate_verdict"] in {"DISCRETE", "WEAK", "CONTINUOUS"}


def test_the_curve_keeps_the_per_k_detail_the_summary_collapses():
    scored = score_space(_space(separation=6.0), run_null_gate=True, **_FAST)
    curve = scored["curve"]
    assert set(curve["k"]) == {2, 3, 4, 5}
    for column in ("prediction_strength_mean", "mean_ari", "null_ps_p95"):
        assert column in curve.columns


def test_the_gate_subset_is_the_declared_one():
    # Gated sweeps run on the primary arm; the two where the arms disagree run on
    # both; everything else is ungated on both.
    assert runs_null_gate("separation-2.0-s0", PRIMARY_ARM)
    assert not runs_null_gate("separation-2.0-s0", "raw")

    for sweep in GATE_BOTH_ARMS:
        assert runs_null_gate(f"{sweep}-0.6-s0", "raw")
        assert runs_null_gate(f"{sweep}-0.6-s0", PRIMARY_ARM)

    for cell_id in ("n-dims-768-s0", "confound-radial-8.0-s0", "coupling-2.0-s0"):
        assert not runs_null_gate(cell_id, PRIMARY_ARM)
        assert not runs_null_gate(cell_id, "raw")
