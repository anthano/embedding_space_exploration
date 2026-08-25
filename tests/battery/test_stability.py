import numpy as np

from embedding_space_exploration.battery.stability import stability_sweep


def test_separated_blobs_are_stable_at_the_true_k():
    # Three well-separated blobs -> the k=3 partition re-emerges under every
    # bootstrap, so mean ARI is near 1 and far above the mis-specified k=2.
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype="float64")
    matrix = np.repeat(centers, 40, axis=0) + rng.standard_normal((120, 2)) * 0.25

    sweep = stability_sweep(matrix, k_values=(2, 3), n_boot=10).set_index("k")
    strong = 0.85
    assert sweep.loc[3, "mean_ari"] > strong
    assert sweep.loc[3, "mean_ari"] > sweep.loc[2, "mean_ari"]


def test_unstructured_blob_is_less_stable_than_real_structure():
    # A single isotropic Gaussian has no reproducible partition; its ARI must be well
    # below that of genuinely separated structure at the same k.
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((200, 4))

    centers = np.array([[0, 0, 0, 0], [8, 0, 0, 0]], dtype="float64")
    structured = np.repeat(centers, 100, axis=0) + rng.standard_normal((200, 4)) * 0.2

    noise_ari = stability_sweep(noise, k_values=(2,), n_boot=10)["mean_ari"].iloc[0]
    struct_ari = stability_sweep(structured, k_values=(2,), n_boot=10)["mean_ari"].iloc[
        0
    ]
    assert struct_ari > noise_ari
