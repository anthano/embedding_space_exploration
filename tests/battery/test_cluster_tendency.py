import numpy as np
import pandas as pd

from embedding_space_exploration.battery.cluster_tendency import (
    cluster_tendency_vs_null,
    null_gate_verdict,
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


def test_null_gate_verdict_requires_null_and_threshold():
    # k=2 beats the null but is below threshold (fragile); k=3 beats neither.
    tendency = pd.DataFrame(
        {
            "k": [2, 3],
            "prediction_strength": [0.72, 0.90],
            "null_ps_median": [0.60, 0.88],
            "null_ps_p95": [0.65, 0.95],
            "exceeds_null": [True, False],
        }
    )
    verdict = null_gate_verdict(tendency, threshold=0.8).iloc[0]
    assert verdict["any_k_beats_null"]
    assert np.isnan(verdict["largest_k_beats_null_and_threshold"])
    assert verdict["verdict"].startswith("WEAK")
