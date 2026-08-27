import numpy as np
import pandas as pd
import pytest

from embedding_space_exploration.battery.cluster_tendency import (
    cluster_tendency_vs_null,
    null_gate_verdict,
    null_margin,
    verdict_label,
)

# Small, fast sweep settings for the tests (the defaults are a heavy one-time gate).
_FAST = {"n_draws": 5, "k_values": (2, 3, 4), "n_repeats": 5, "random_state": 0}


def test_real_structure_beats_the_null():
    # Three well-separated blobs: real prediction strength at k=3 is ~1 while a
    # covariance-matched Gaussian (one blob) cannot reproduce it -> exceeds_null.
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [12, 0], [0, 12]], dtype="float64")
    matrix = np.repeat(centers, 40, axis=0) + rng.standard_normal((120, 2)) * 0.3

    tendency = cluster_tendency_vs_null(matrix, **_FAST)
    verdict = null_gate_verdict(tendency).iloc[0]

    assert tendency.set_index("k").loc[3, "exceeds_null"]
    assert verdict["any_k_beats_null"]
    assert verdict["verdict"].startswith("DISCRETE")


def test_unstructured_blob_is_continuous():
    # A single isotropic Gaussian has no discrete structure: the real sweep should
    # sit inside its own covariance-matched null band, so no k is believed.
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((200, 5))

    tendency = cluster_tendency_vs_null(matrix, **_FAST)
    verdict = null_gate_verdict(tendency).iloc[0]

    assert not tendency["exceeds_null"].any()
    assert verdict["verdict"].startswith("CONTINUOUS")


def test_resphere_null_keeps_the_spherical_arm_on_the_shell():
    # Three separated blobs projected onto the unit sphere. With resphere_null the
    # null draws are L2-normalised onto the same shell, so the comparison is on
    # -manifold and real angular structure still clears the null at k=3.
    rng = np.random.default_rng(2)
    centers = np.array([[3, 0], [-3, 1], [0, 3]], dtype="float64")
    matrix = np.repeat(centers, 40, axis=0) + rng.standard_normal((120, 2)) * 0.2
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)  # onto the unit circle

    tendency = cluster_tendency_vs_null(matrix, resphere_null=True, **_FAST)
    assert tendency.set_index("k").loc[3, "exceeds_null"]


def test_the_margin_is_the_share_of_headroom_above_the_null():
    # 0.769 against a null median of 0.389 leaves 0.611 of headroom, of which
    # 0.380 is captured -> 0.622. Bounded by 1 at every k, which is what makes
    # k=4 and k=10 comparable at all.
    tendency = pd.DataFrame(
        {
            "k": [4, 10],
            "prediction_strength": [0.7691, 0.1701],
            "null_ps_median": [0.3887, 0.1531],
            "null_ps_p95": [0.4721, 0.1592],
            "exceeds_null": [True, True],
        }
    )
    margin = null_margin(tendency).set_index("k")["headroom_margin"]
    assert margin.loc[4] == pytest.approx(0.622, abs=0.001)
    assert margin.loc[10] == pytest.approx(0.020, abs=0.001)


def test_the_verdict_reads_magnitude_not_the_largest_k_that_clears_a_bar():
    # The failure this replaces: `exceeds_null` is near-automatic in the high-k
    # tail, so a "largest k" rule lands on the weakest evidence in the table.
    # k=4 carries the structure; k=10 is 2% above chance.
    tendency = pd.DataFrame(
        {
            "k": [4, 10],
            "prediction_strength": [0.7691, 0.1701],
            "null_ps_median": [0.3887, 0.1531],
            "null_ps_p95": [0.4721, 0.1592],
            "exceeds_null": [True, True],
        }
    )
    verdict = null_gate_verdict(tendency, threshold=0.25).iloc[0]
    assert verdict["verdict"].startswith("DISCRETE")
    assert verdict["k_at_max_margin"] == 4
    assert verdict["max_headroom_margin"] == pytest.approx(0.622, abs=0.001)


def test_beating_the_null_by_a_hair_is_weak_not_discrete():
    tendency = pd.DataFrame(
        {
            "k": [2, 3],
            "prediction_strength": [0.62, 0.90],
            "null_ps_median": [0.60, 0.92],
            "null_ps_p95": [0.61, 0.95],
            "exceeds_null": [True, False],
        }
    )
    verdict = null_gate_verdict(tendency, threshold=0.25).iloc[0]
    assert verdict["any_k_beats_null"]
    assert verdict["verdict"].startswith("WEAK")


def test_the_label_is_a_function_of_the_recorded_margin_alone():
    # The point of splitting this out: a finished run records the margin, so the
    # label can be recomputed later under a revised threshold. Same margin, two
    # thresholds, two labels -- no re-run.
    assert verdict_label(0.212, beats_null=True, threshold=0.10) == "DISCRETE"
    assert verdict_label(0.212, beats_null=True, threshold=0.25) == "WEAK"


def test_failing_the_null_outranks_the_margin():
    # A large margin means nothing if no k cleared the null band at all.
    assert verdict_label(0.9, beats_null=False, threshold=0.10) == "CONTINUOUS"


def test_the_sentence_and_the_label_cannot_disagree():
    # `null_gate_verdict` builds its prose from `verdict_label`, so the two can
    # never drift apart -- which is what lets the tables derive one from the
    # margin while the harness stores the other.
    tendency = pd.DataFrame(
        {
            "k": [4, 10],
            "prediction_strength": [0.7691, 0.1701],
            "null_ps_median": [0.3887, 0.1531],
            "null_ps_p95": [0.4721, 0.1592],
            "exceeds_null": [True, True],
        }
    )
    for threshold in (0.10, 0.25, 0.90):
        verdict = null_gate_verdict(tendency, threshold=threshold).iloc[0]
        label = verdict_label(
            verdict["max_headroom_margin"],
            beats_null=verdict["any_k_beats_null"],
            threshold=threshold,
        )
        assert verdict["verdict"].startswith(label)
