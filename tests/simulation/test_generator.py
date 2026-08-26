import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.neighbors import KNeighborsRegressor

from embedding_space_exploration.battery.check_embeddings import embedding_checks
from embedding_space_exploration.battery.confound_orientation import (
    principal_component_regression,
)
from embedding_space_exploration.simulation.generator import (
    simulate_embeddings,
    simulate_split,
)


def _matrix(space):
    dims = [c for c in space["embeddings"].columns if c.startswith("dim_")]
    return space["embeddings"][dims].to_numpy()


def test_frames_share_one_person_id_order():
    space = simulate_embeddings(n_patients=120, n_dims=16, intrinsic_dim=6)
    person_id = space["embeddings"]["person_id"].to_numpy()
    for key in ("covariates", "truth", "split"):
        assert np.array_equal(space[key]["person_id"].to_numpy(), person_id)
    assert _matrix(space).shape == (120, 16)
    assert space["covariates"].columns.tolist() == ["person_id", "log_n_events"]
    assert len(space["spec"]) == 1


def test_planted_clusters_are_recoverable_and_absent_when_not_planted():
    # The core claim: `separation` plants structure a check can find, and
    # `separation=0` plants none. Anything reading ARI ~1 on the second is broken.
    common = {"n_patients": 600, "n_dims": 16, "intrinsic_dim": 6, "n_clusters": 3}
    planted = simulate_embeddings(separation=6.0, **common)
    unstructured = simulate_embeddings(separation=0.0, **common)

    def ari(space):
        labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(
            _matrix(space)
        )
        return adjusted_rand_score(space["truth"]["cluster"].to_numpy(), labels)

    assert ari(planted) > 0.95
    assert abs(ari(unstructured)) < 0.05


def test_intrinsic_dimension_is_the_exact_noiseless_rank():
    space = simulate_embeddings(
        n_patients=300, n_dims=32, intrinsic_dim=5, n_clusters=3, noise=0.0
    )
    assert np.linalg.matrix_rank(_matrix(space)) == 5

    # Ambient noise fills every direction, so the *numerical* rank goes full and
    # only an effective-rank measure still reports the planted dimensionality.
    blurred = simulate_embeddings(
        n_patients=300, n_dims=32, intrinsic_dim=5, n_clusters=3, noise=0.3
    )
    assert np.linalg.matrix_rank(_matrix(blurred)) == 32


def test_anisotropy_is_declared_in_the_units_the_check_reports():
    for target in (0.0, 0.5, 0.9):
        space = simulate_embeddings(
            n_patients=500,
            n_dims=16,
            intrinsic_dim=6,
            separation=0.0,
            anisotropy=target,
        )
        measured = embedding_checks(space["embeddings"])["mean_cosine_to_centroid"]
        assert measured.iloc[0] == pytest.approx(target, abs=0.05)


@pytest.mark.parametrize("structure", ["clusters", "continuum"])
def test_planting_structure_does_not_smuggle_in_a_cone(structure):
    # The knobs have to move independently or nothing downstream can be
    # attributed: with `anisotropy=0` the space stays isotropic however far
    # apart the structure is placed, so A4 reads the cone and only the cone.
    space = simulate_embeddings(
        n_patients=500,
        n_dims=32,
        intrinsic_dim=6,
        n_clusters=3,
        separation=10.0,
        structure=structure,
    )
    measured = embedding_checks(space["embeddings"])["mean_cosine_to_centroid"]
    assert abs(measured.iloc[0]) < 0.1


def test_axis_confound_lands_on_the_leading_component():
    # (a) of the dissociation: written along one direction, strength above every
    # structural SD, so PCR sees it on PC0 -- the confound C1 must catch.
    space = simulate_embeddings(
        n_patients=600,
        n_dims=16,
        intrinsic_dim=6,
        n_clusters=3,
        separation=2.0,
        confound_orientation="axis",
        confound_strength=8.0,
    )
    per_covariate = principal_component_regression(
        space["embeddings"], space["covariates"]
    )["per_covariate"].set_index("covariate")
    assert per_covariate.loc["log_n_events", "leading_pc_r2"] > 0.9


def test_radial_confound_is_decodable_but_aligned_to_no_component():
    # (b) of the dissociation, and the reason the nonlinear probe exists: the
    # confound is fully recoverable from the space, yet no linear direction --
    # and so no principal component -- carries it.
    space = simulate_embeddings(
        n_patients=800,
        n_dims=16,
        intrinsic_dim=3,
        n_clusters=1,
        separation=0.0,
        confound_orientation="radial",
        confound_strength=4.0,
        noise=0.05,
    )
    pcr = principal_component_regression(space["embeddings"], space["covariates"])
    assert pcr["per_pc"]["r2__log_n_events"].max() < 0.05
    assert pcr["per_covariate"].loc[0, "variance_weighted_r2"] < 0.05

    matrix, confound = _matrix(space), space["covariates"]["log_n_events"].to_numpy()
    train, test = slice(None, 600), slice(600, None)
    linear = LinearRegression().fit(matrix[train], confound[train])
    nonlinear = KNeighborsRegressor(n_neighbors=10).fit(matrix[train], confound[train])

    assert r2_score(confound[test], linear.predict(matrix[test])) < 0.05
    assert r2_score(confound[test], nonlinear.predict(matrix[test])) > 0.5


def test_confound_coupling_entangles_the_confound_with_the_signal():
    # The case where a correction cannot be free: the confound carries cluster
    # identity, so removing it removes signal too (what C4's arms are scored on).
    common = {
        "n_patients": 600,
        "n_dims": 16,
        "intrinsic_dim": 6,
        "n_clusters": 3,
        "confound_orientation": "axis",
        "confound_strength": 2.0,
    }
    independent = simulate_embeddings(confound_cluster_coupling=0.0, **common)
    coupled = simulate_embeddings(confound_cluster_coupling=3.0, **common)

    def spread_of_cluster_means(space):
        frame = space["covariates"].join(space["truth"]["cluster"])
        return frame.groupby("cluster")["log_n_events"].mean().std()

    assert spread_of_cluster_means(independent) < 0.2
    assert spread_of_cluster_means(coupled) > 1.5


def test_continuum_fills_the_gaps_between_the_same_centres():
    # D6's comparator: identical waypoints and identical local scatter, but the
    # points spread along the polyline instead of piling up at its corners.
    common = {
        "n_patients": 600,
        "n_dims": 16,
        "intrinsic_dim": 6,
        "n_clusters": 3,
        "separation": 8.0,
    }
    blobs = simulate_embeddings(structure="clusters", **common)
    filament = simulate_embeddings(structure="continuum", **common)

    # Both arms are measured against the *same* centres -- the blob centroids.
    centroids = (
        KMeans(n_clusters=3, n_init=10, random_state=0)
        .fit(_matrix(blobs))
        .cluster_centers_
    )

    def distance_to_nearest_centre(space):
        distances = np.linalg.norm(
            _matrix(space)[:, None, :] - centroids[None, :, :], axis=2
        )
        return distances.min(axis=1).mean()

    assert distance_to_nearest_centre(filament) > 2 * distance_to_nearest_centre(blobs)
    assert filament["truth"]["cluster"].isna().all()
    assert filament["truth"]["position"].between(0, 1).all()
    assert blobs["truth"]["position"].isna().all()


def test_decoy_covariates_are_pure_noise():
    space = simulate_embeddings(
        n_patients=400, n_dims=16, intrinsic_dim=6, n_decoy_covariates=2
    )
    assert space["covariates"].columns.tolist() == [
        "person_id",
        "log_n_events",
        "decoy_0",
        "decoy_1",
    ]
    per_covariate = principal_component_regression(
        space["embeddings"], space["covariates"]
    )["per_covariate"].set_index("covariate")
    assert per_covariate.loc["decoy_0", "variance_weighted_r2"] < 0.05


def test_the_same_seed_gives_the_same_space():
    kwargs = {"n_patients": 200, "n_dims": 16, "intrinsic_dim": 6, "random_state": 7}
    first, second = simulate_embeddings(**kwargs), simulate_embeddings(**kwargs)
    pd.testing.assert_frame_equal(first["embeddings"], second["embeddings"])
    pd.testing.assert_frame_equal(first["truth"], second["truth"])

    other = simulate_embeddings(**{**kwargs, "random_state": 8})
    assert not np.allclose(_matrix(first), _matrix(other))


def test_split_covers_every_patient_once_in_the_declared_proportions():
    split = simulate_split(np.arange(1000))
    counts = split["split"].value_counts()
    assert len(split) == 1000
    assert counts["train"] == 600
    assert counts["val"] == 200
    assert counts["test"] == 200


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_dims": 4, "intrinsic_dim": 8}, "cannot hold"),
        ({"intrinsic_dim": 2, "n_clusters": 4}, "too small"),
        ({"structure": "continuum", "n_clusters": 1}, "at least two waypoints"),
        ({"structure": "blobs"}, "must be 'clusters' or 'continuum'"),
        ({"confound_orientation": "diagonal"}, "confound_orientation must be"),
        ({"anisotropy": 1.0}, "must be in .0, 1."),
        ({"n_clusters": 3, "cluster_proportions": (0.5, 0.5)}, "summing to 1"),
    ],
)
def test_unbuildable_spaces_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        simulate_embeddings(n_patients=50, **kwargs)
