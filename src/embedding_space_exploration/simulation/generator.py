"""Synthetic embedding matrices with ground truth -- the Tier 0 calibration data.

One generator, four independently controllable knobs, matching the four things
Tier 0 has to calibrate the battery against:

- known cluster structure -- ``n_clusters`` / ``separation`` /
  ``cluster_proportions``, and ``structure="continuum"`` for the case that has
  no discrete structure but passes for it. Ground truth: the ``cluster``
  column of the ``truth`` frame (D2/D3/D4 and the D1 null gate score against it).
- known confound loading -- ``confound_strength`` sets how strongly a
  nuisance is written into the space, ``confound_orientation`` sets how, and
  ``confound_cluster_coupling`` sets how far it is entangled with the signal
  (C1, C4 and the new nonlinear decodability probe).
- known anisotropy -- ``anisotropy`` is declared in the units A4 reports it
  in (mean cosine to the centroid), so "did the check recover it?" is a
  subtraction.
- known intrinsic dimension -- ``intrinsic_dim`` is the exact rank of the
  noiseless matrix, sunk into an ``n_dims``-dimensional ambient space and then
  blurred by ``noise``. RankMe (A5) is vision-calibrated and unvalidated for EHR
  clustering, so here we can ask it to report the truth and see how far off it is.

The orientation/decodability dissociation: ``confound_orientation`` is the
knob the paper's thesis-in-miniature needs, and the two settings are opposite:

- ``"axis"`` writes the confound along a single latent direction, so PCA sees it
  as a component. Push ``confound_strength`` above the largest structural SD
  (``sqrt(1 + separation**2 * p * (1 - p) / 2)`` for a cluster axis) and it
  *becomes* PC0 -- the confound C1 must catch.
- ``"radial"`` writes it as a **radius within a 2-D shell at a random angle**:
  the patient's distance from the origin in that plane is a strictly monotone
  function of the confound, but the direction is uniform and independent of it.
  Every fixed direction therefore has zero linear association with the confound
  in expectation -- C1's per-PC R^2 goes to 0 -- while a nonlinear learner
  recovers it exactly from the squared norm. *Fully decodable, aligned to no
  component*: the confound C1 must miss.

Ground truth lives in the ``truth`` frame and the knobs in the one-row ``spec``
frame; neither is ever handed to a check. Pure functions, no I/O -- the harness
that sweeps this and scores the battery against it is a separate module.
"""

import numpy as np
import pandas as pd

from embedding_space_exploration.config import RANDOM_STATE
from embedding_space_exploration.data_management.splits import TEST, TRAIN, VAL


def simulate_embeddings(
    *,
    n_patients=2_000,
    n_dims=64,
    intrinsic_dim=8,
    structure="clusters",
    n_clusters=4,
    separation=3.0,
    cluster_proportions=None,
    confound_orientation="none",
    confound_strength=0.0,
    confound_cluster_coupling=0.0,
    confound_column="log_n_events",
    n_decoy_covariates=0,
    noise=0.1,
    anisotropy=0.0,
    random_state=RANDOM_STATE,
):
    """Draw a synthetic embedding space with planted, recorded structure.

    The space is built in a low-dimensional latent space and then sunk into the
    ambient one, in three blocks that never interfere with each other:
    ``[signal | confound | background]``, summing to ``intrinsic_dim``. The
    latent is projected through an orthonormal basis (so the noiseless matrix has
    rank exactly ``intrinsic_dim``), blurred by isotropic ambient ``noise``, and
    finally pushed off the origin into a cone of the requested ``anisotropy``.

    Because the projection is orthonormal and the within-cluster scatter is unit
    normal, ``separation`` is a distance in units of within-cluster SD, and the
    centres are placed mutually equidistant and about the origin -- so it is the
    *only* thing that sets how far apart the blobs are, and turning it up does
    not smuggle in a cone. Ambient noise inflates the scatter, so what a check
    actually faces is ``separation / sqrt(1 + noise**2)``, recorded as
    ``effective_separation``.

    Args:
        n_patients: Rows.
        n_dims: Ambient dimensionality (the width of the embedding a model would
            emit). Must be at least ``intrinsic_dim``.
        intrinsic_dim: Rank of the noiseless matrix. Must be large enough to hold
            the signal and confound blocks; whatever is left over becomes
            isotropic unit-variance background.
        structure: ``"clusters"`` for ``n_clusters`` Gaussian blobs, or
            ``"continuum"`` for points spread uniformly along the polyline
            through those same centres -- identical extent and local scatter,
            no discrete structure. The comparator D6 needs, and the case that
            can pass prediction strength, the null gate and outcome separation
            while being a sliced continuum.
        n_clusters: Number of planted clusters (or continuum waypoints).
        separation: Distance between any two cluster centres, in units of
            within-cluster SD -- so ``3`` means centres three within-cluster SDs
            apart, and ``0`` collapses every cluster onto one blob, the
            unstructured case the D1 null gate must return ``CONTINUOUS`` on.
        cluster_proportions: Mixing proportions, length ``n_clusters``, summing
            to 1. Defaults to balanced. Unbalance it to ask whether a check finds
            a rare cluster.
        confound_orientation: ``"none"``, ``"axis"`` (one linear direction, PCA
            sees it) or ``"radial"`` (decodable but aligned to no component).
            See the module docstring -- the ``"axis"``/``"radial"`` contrast is
            the dissociation the nonlinear probe exists to catch.
        confound_strength: How strongly the confound is written into the space.
            In ``"axis"`` mode this is the SD of its contribution, so it is
            directly comparable with the structural SDs above.
        confound_cluster_coupling: Per-cluster mean shift of the confound (in
            confound SDs, spread over ``linspace(-1, 1, n_clusters)``; against
            ``position`` for a continuum). ``0`` leaves the confound independent
            of the signal -- the easy case, where removing it costs nothing. Turn
            it up to build the case C4's drop and residual arms can only damage.
        confound_column: Name of the confound in the covariate frame. Defaults
            to the real analysis's dominant nuisance so the frame is drop-in for
            ``principal_component_regression`` and ``residualize_embeddings``.
        n_decoy_covariates: Extra pure-noise covariates. They load on nothing, so
            what a check reports for them is its false-positive floor.
        noise: SD of isotropic ambient noise, added in every one of ``n_dims``
            directions. This is what separates *intrinsic* from *numerical*
            rank: above zero the matrix is full-rank and only its **effective**
            rank still reports ``intrinsic_dim``.
        anisotropy: Target mean cosine to the centroid, in ``[0, 1)``. Achieved
            by offsetting every row along one shared direction by
            ``mean_row_norm * a / sqrt(1 - a**2)``; exact up to the Jensen gap of
            the row-norm average, so treat it as accurate to a few hundredths.
        random_state: Seed. Everything below is drawn from it.

    Returns:
        A dict of five frames, all in the same ``person_id`` row order:

        - ``embeddings``: ``person_id`` plus ``dim_0 .. dim_N`` -- the battery's
          input contract.
        - ``covariates``: ``person_id``, the confound, and any decoys -- the
          ``confound_orientation`` / ``prep.residualize_embeddings`` contract.
        - ``truth``: ``person_id``, ``cluster`` (``Int64``, ``<NA>`` for a
          continuum), ``position`` (float, ``NaN`` for clusters). What no check
          is allowed to see.
        - ``split``: ``person_id``, ``split`` -- a 60/20/20 draw under the
          ``splits`` role names, so ``fit_mask`` works unchanged.
        - ``spec``: one row of every knob plus the derived layout, so a sweep is
          a ``concat`` of spec rows joined to the measurements.

    Raises:
        ValueError: If the requested space is not constructible -- ambient
            narrower than intrinsic, a latent budget too small for the signal and
            confound blocks, a continuum through fewer than two waypoints,
            proportions that do not sum to 1, or an unreachable anisotropy.
    """
    n_signal, n_confound, n_background = _latent_layout(
        structure=structure,
        n_clusters=n_clusters,
        confound_orientation=confound_orientation,
        intrinsic_dim=intrinsic_dim,
    )
    _validate(
        n_dims=n_dims,
        intrinsic_dim=intrinsic_dim,
        anisotropy=anisotropy,
        n_clusters=n_clusters,
        cluster_proportions=cluster_proportions,
    )

    rng = np.random.default_rng(random_state)
    centres = _cluster_centres(n_signal, separation, cluster_proportions)
    labels, position, signal = _signal_block(
        rng,
        n_patients=n_patients,
        structure=structure,
        centres=centres,
        proportions=cluster_proportions,
    )
    confound = _confound_values(
        rng,
        n_patients=n_patients,
        labels=labels,
        position=position,
        coupling=confound_cluster_coupling,
        n_clusters=n_signal,
    )
    latent = np.column_stack(
        [
            signal,
            _confound_block(
                rng,
                confound=confound,
                orientation=confound_orientation,
                strength=confound_strength,
                n_confound=n_confound,
            ),
            rng.standard_normal((n_patients, n_background)),
        ]
    )

    matrix = latent @ _orthonormal_basis(rng, n_dims, intrinsic_dim).T
    matrix = matrix + noise * rng.standard_normal(matrix.shape)
    matrix, cone_norm = _add_cone(rng, matrix, anisotropy)

    person_id = np.arange(n_patients)
    return {
        "embeddings": _embedding_frame(person_id, matrix),
        "covariates": _covariate_frame(
            rng, person_id, confound, confound_column, n_decoy_covariates
        ),
        "truth": _truth_frame(person_id, labels, position),
        "split": simulate_split(person_id, random_state=random_state),
        "spec": pd.DataFrame(
            [
                {
                    "n_patients": n_patients,
                    "n_dims": n_dims,
                    "intrinsic_dim": intrinsic_dim,
                    "structure": structure,
                    "n_clusters": n_clusters,
                    "separation": separation,
                    "effective_separation": separation / np.sqrt(1 + noise**2),
                    "confound_orientation": confound_orientation,
                    "confound_strength": confound_strength,
                    "confound_cluster_coupling": confound_cluster_coupling,
                    "n_decoy_covariates": n_decoy_covariates,
                    "noise": noise,
                    "anisotropy": anisotropy,
                    "cone_norm": cone_norm,
                    "n_signal_dims": n_signal,
                    "n_confound_dims": n_confound,
                    "n_background_dims": n_background,
                    "random_state": random_state,
                }
            ]
        ),
    }


def simulate_split(person_id, *, fractions=(0.6, 0.2, 0.2), random_state=RANDOM_STATE):
    """Assign simulated patients to the three ``splits`` roles at random.

    Tier 1 reads EHRSHOT's shipped assignment and Tier 0 has none to read, so it
    draws its own. 60/20/20 because the roles need a large fit set, and val and
    test need enough rows to separate a selection from a claim.

    Args:
        person_id: Array of person ids.
        fractions: ``(train, val, test)`` shares, summing to 1.
        random_state: Seed for the permutation.

    Returns:
        Frame with ``person_id`` and ``split``, in input order.
    """
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(person_id))
    cuts = np.cumsum(np.round(np.asarray(fractions) * len(person_id)).astype(int))[:2]

    labels = np.empty(len(person_id), dtype=object)
    for role, rows in zip((TRAIN, VAL, TEST), np.split(order, cuts), strict=True):
        labels[rows] = role
    return pd.DataFrame({"person_id": person_id, "split": labels})


# ======================================================================================
# HELPER FUNCTIONS AND CONSTANTS
# ======================================================================================

# Latent dimensions each confound orientation consumes: a single direction for the
# linear one, a plane for the radial one (a radius needs somewhere to point).
_CONFOUND_DIMS = {"none": 0, "axis": 1, "radial": 2}

# Log-scale spread of the radial confound: the shell radius is
# `strength * exp(_RADIAL_SPREAD * confound)`. Strictly positive and strictly
# monotone, so the confound is recoverable everywhere -- no fold, unlike a signed
# radius -- and it makes the covariate a log of the thing it drives, which is what
# `log_n_events` is. 0.5 gives a ~1.8x interquartile spread in radius: plainly
# decodable without heavy enough tails to distort the PCA.
_RADIAL_SPREAD = 0.5


def _latent_layout(*, structure, n_clusters, confound_orientation, intrinsic_dim):
    """Split the latent budget into (signal, confound, background) dimensions."""
    if structure not in ("clusters", "continuum"):
        raise ValueError(
            f"structure must be 'clusters' or 'continuum', got {structure}"
        )
    if confound_orientation not in _CONFOUND_DIMS:
        raise ValueError(
            f"confound_orientation must be one of {sorted(_CONFOUND_DIMS)}, "
            f"got {confound_orientation}"
        )
    if structure == "continuum" and n_clusters < 2:
        raise ValueError("a continuum needs at least two waypoints (n_clusters >= 2)")

    # One signal dimension per centre, in both modes: the continuum runs along the
    # polyline through the same centres, so the two arms differ only in whether the
    # space between them is filled.
    n_signal = n_clusters
    n_confound = _CONFOUND_DIMS[confound_orientation]
    n_background = intrinsic_dim - n_signal - n_confound
    if n_background < 0:
        raise ValueError(
            f"intrinsic_dim={intrinsic_dim} is too small: the signal block needs "
            f"{n_signal} dimensions and the confound block {n_confound}"
        )
    return n_signal, n_confound, n_background


def _validate(*, n_dims, intrinsic_dim, anisotropy, n_clusters, cluster_proportions):
    """Reject the parameter combinations that cannot be built."""
    if n_dims < intrinsic_dim:
        raise ValueError(
            f"n_dims={n_dims} cannot hold an intrinsic_dim={intrinsic_dim} subspace"
        )
    if not 0 <= anisotropy < 1:
        raise ValueError(
            f"anisotropy is a mean cosine and must be in [0, 1), got {anisotropy}"
        )
    if cluster_proportions is None:
        return
    proportions = np.asarray(cluster_proportions, dtype="float64")
    if len(proportions) != n_clusters or not np.isclose(proportions.sum(), 1.0):
        raise ValueError(
            f"cluster_proportions must have {n_clusters} entries summing to 1, "
            f"got {cluster_proportions}"
        )


def _cluster_centres(n_clusters, separation, proportions):
    """Mutually equidistant cluster centres, centred on the origin.

    One centre per signal axis, scaled so any two are exactly ``separation``
    apart, then shifted by the mixture mean so the signal block has mean zero.
    Without that shift every centre would sit in the positive orthant and the
    cloud would acquire a cone of its own -- ``separation`` would silently move
    the anisotropy the ``anisotropy`` knob is supposed to own, and the two could
    not be calibrated apart.
    """
    weights = (
        np.full(n_clusters, 1 / n_clusters)
        if proportions is None
        else np.asarray(proportions, dtype="float64")
    )
    return (separation / np.sqrt(2)) * (np.eye(n_clusters) - weights)


def _signal_block(rng, *, n_patients, structure, centres, proportions):
    """Draw the structural latent block and the ground truth that describes it."""
    n_signal = centres.shape[1]
    scatter = rng.standard_normal((n_patients, n_signal))

    if structure == "clusters":
        labels = rng.choice(n_signal, size=n_patients, p=proportions)
        position = np.full(n_patients, np.nan)
        return labels, position, centres[labels] + scatter

    # Uniform along the polyline through the centres: same endpoints, same extent
    # and same local scatter as the blobs, with the gaps filled in.
    position = rng.uniform(0.0, 1.0, n_patients)
    travelled = position * (n_signal - 1)
    lower = np.minimum(np.floor(travelled).astype(int), n_signal - 2)
    frac = (travelled - lower)[:, None]
    means = centres[lower] * (1 - frac) + centres[lower + 1] * frac

    # The walk's own mean is not the centres' mean -- interior waypoints are
    # traversed twice -- so recentre on it. The blob and continuum arms are only
    # a like-for-like comparison if neither carries a cone the other lacks.
    weights = np.full(n_signal, 1.0 / (n_signal - 1))
    weights[0] = weights[-1] = 0.5 / (n_signal - 1)
    return np.full(n_patients, np.nan), position, means - weights @ centres + scatter


def _confound_values(rng, *, n_patients, labels, position, coupling, n_clusters):
    """Draw the confound, optionally shifted by the structure it is entangled with."""
    confound = rng.standard_normal(n_patients)
    if coupling == 0:
        return confound
    if np.isnan(position).all():
        shift = np.linspace(-1.0, 1.0, n_clusters)[labels]
    else:
        shift = 2.0 * position - 1.0
    return confound + coupling * shift


def _confound_block(rng, *, confound, orientation, strength, n_confound):
    """Write the confound into its latent dimensions, linearly or radially."""
    if n_confound == 0:
        return np.zeros((len(confound), 0))
    if orientation == "axis":
        # One direction, so it is a principal component -- a leading one once
        # `strength` exceeds the largest structural SD.
        return (strength * confound)[:, None]

    # Radial: distance from the origin in a plane carries the confound, the angle
    # is uniform and independent of it. Every fixed direction is therefore
    # uncorrelated with the confound in expectation -- invisible to PCR and to any
    # linear probe -- while the squared norm recovers it exactly.
    radius = strength * np.exp(_RADIAL_SPREAD * confound)
    angle = rng.uniform(0.0, 2 * np.pi, len(confound))
    return radius[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])


def _orthonormal_basis(rng, n_dims, intrinsic_dim):
    """An ``(n_dims, intrinsic_dim)`` basis with orthonormal columns.

    Orthonormal so the projection is an isometry: within-cluster scatter stays
    unit, ``separation`` keeps its declared units, and the noiseless matrix has
    rank exactly ``intrinsic_dim``.
    """
    basis, _ = np.linalg.qr(rng.standard_normal((n_dims, intrinsic_dim)))
    return basis


def _add_cone(rng, matrix, anisotropy):
    """Offset every row along one shared direction to hit a target mean cosine.

    For rows ``y`` scattered about the origin with mean norm ``r``, adding a
    shared ``mu`` of norm ``m`` gives a mean cosine to the centroid of about
    ``m / sqrt(r**2 + m**2)``; inverting that for ``m`` puts the requested
    anisotropy in the units A4 reports.
    """
    if anisotropy == 0:
        return matrix, 0.0
    radius = float(np.linalg.norm(matrix, axis=1).mean())
    cone_norm = radius * anisotropy / np.sqrt(1 - anisotropy**2)
    direction = rng.standard_normal(matrix.shape[1])
    direction = direction / np.linalg.norm(direction)
    return matrix + cone_norm * direction, cone_norm


def _embedding_frame(person_id, matrix):
    """Matrix to the battery's ``person_id`` + ``dim_*`` input contract."""
    frame = pd.DataFrame(matrix, columns=[f"dim_{i}" for i in range(matrix.shape[1])])
    frame.insert(0, "person_id", person_id)
    return frame


def _covariate_frame(rng, person_id, confound, confound_column, n_decoys):
    """The confound plus any pure-noise decoys, as a covariate frame."""
    frame = pd.DataFrame({"person_id": person_id, confound_column: confound})
    for i in range(n_decoys):
        frame[f"decoy_{i}"] = rng.standard_normal(len(person_id))
    return frame


def _truth_frame(person_id, labels, position):
    """The planted structure, held back from every check."""
    return pd.DataFrame(
        {
            "person_id": person_id,
            "cluster": pd.Series(labels, dtype="float64").astype("Int64"),
            "position": position,
        }
    )
