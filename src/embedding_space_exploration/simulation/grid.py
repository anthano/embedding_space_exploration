"""The Tier 0 calibration design: which spaces to build, and what to measure on them.

Kept apart from the task files on purpose. ``@pytask.task`` registers tasks in a
global registry the moment a module is imported, so anything that imports a task
module to read a constant silently adds the whole DAG to every later
``pytask.build()`` in that process -- which is how a scoped build ends up
collecting 138 simulated cells. The design is data; only the wiring is a task.

``generator.py`` can build any space; the grid
below declares which ones we actually evaluate, and every sweep exists to
answer one question from the Tier 0 checklist:

- ``separation`` (0 -> 6) -- **how much structure does it take to beat the weak
  null?** The headline curve. D1's null is a single covariance-matched Gaussian
  and the battery itself calls it weak; this is where we find out how weak.
  ``separation=0`` is the false-positive end, and any check that fires there is
  broken.
- ``continuum`` (2 -> 6) -- **D6.** The same waypoints, filled in. Prediction
  strength, the null gate and outcome separation can all pass for a sliced
  continuum; this says at what separation they start to.
- ``anisotropy`` (0.3 -> 0.9) -- **A4**, and what a cone costs every check
  downstream of it. Structure is present here, so a verdict cannot be
  attributed to the cone -- that is what ``cone-only`` is for.
- ``cone-only`` (0.3 -> 0.9 at ``separation=0``) -- **the Corpas trap, and the
  sharpest question the null gate faces.** A single anisotropic blob with no
  discrete structure at all: does prediction strength still report a "good"
  k>=2, and does the covariance-matched null catch it? D1 exists precisely
  because the rest of the literature does not ask this. If the gate fails
  anywhere, the most likely place is here.
- ``n-dims`` (28 -> 768) -- **A5.** RankMe is dimension-dependent, and 28 / 256 /
  768 is the real spread across the model grid. It is vision-calibrated and
  unvalidated for EHR clustering; this closes that.
- ``intrinsic-dim`` (4 -> 64) -- A5 from the other side: does effective rank
  track the planted dimensionality?
- ``noise`` (0.25 -> 2.0) -- where intrinsic and numerical rank part company, and
  the SNR at which each check gives up.
- ``confound-axis`` (strength 0.5 -> 8) -- **C1.** At what loading does PCR see a
  nuisance on a leading PC?
- ``confound-radial`` (strength 0.5 -> 8) -- **the dissociation.** The same
  loadings, written so that no component carries them. C1 should stay flat across
  this whole sweep while a nonlinear probe climbs it. That gap is the paper's
  thesis in miniature, and if it does not appear here it will not appear on
  EHRSHOT either.
- ``coupling`` (1 -> 4) -- **C4.** The confound entangled with the signal, where
  the drop and residual arms can only do damage.
- ``imbalance`` (3 shapes) -- does a check find a rare cluster, or only
  equal-sized ones?
- ``n-patients`` (2k / 5k / 10k) -- which metrics are stable at EHRSHOT's n,
  asked here before anything is spent on real data.

Every condition is drawn at ``N_REPLICATES`` seeds, because a calibration table
has to report a band per condition rather than one draw. Every cell also carries
decoy covariates that load on nothing, so each check's false-positive floor comes
for free wherever it is measured.

The grid moves one knob at a time off ``BASE`` and is never a full crossing: ~44
conditions instead of the thousands a product would give, and each one reads as
"this knob, that curve". Where two knobs plausibly interact, the interaction is a
condition in a later sweep, not a dimension of this one.

**Both preprocessing arms, everywhere.** ``raw`` is the honest baseline and
``spherical`` the primary arm, and the arm sits between the space and every
clustering check -- a check's behaviour is a property of the pair, not of the
space. The ``cone-only`` preview already showed the two arms disagreeing on an
anisotropic blob, which is exactly the kind of thing that would be invisible if
only the primary arm were scored.

**The null gate on a declared subset.** D1 costs ``N_NULL_DRAWS + 1`` full
prediction-strength sweeps -- ~90s per cell-arm against ~11s for everything else
combined, so running it everywhere would be ~85% of the compute spent on the
sweeps it says least about. It runs where the gate is the instrument under test
(``GATE_SWEEPS``), on the primary arm, plus both arms where the disagreement
between arms is itself the question (``GATE_BOTH_ARMS``). It does **not** run on
the ``n-dims``, ``intrinsic-dim``, ``confound-*`` or ``coupling`` sweeps: those
calibrate A5 and C1, and the gate sees a PCA(50) projection in which ambient
width has already been washed out.

The gate is run at the frozen constants, never at reduced settings. A gate
calibrated cheaper than it will be used is not a calibration of the gate.

Cost: ~280 cell-arms at ~11s, plus ~96 gate runs at ~90s -- on the order of three
hours single-core, well under an hour parallelised.
"""

from embedding_space_exploration.config import RANDOM_STATE

FRAMES = ("embeddings", "covariates", "truth", "split", "spec")
N_REPLICATES = 3
BASE = {
    "n_patients": 2_000,
    "n_dims": 128,
    "intrinsic_dim": 16,
    "structure": "clusters",
    "n_clusters": 4,
    "separation": 3.0,
    "confound_orientation": "none",
    "confound_strength": 0.0,
    "confound_cluster_coupling": 0.0,
    "n_decoy_covariates": 2,
    "noise": 0.1,
    "anisotropy": 0.0,
}


def calibration_grid():
    """Build the Tier 0 grid: ``{cell_id: simulate_embeddings kwargs}``.

    One knob at a time off ``BASE``, then every condition replicated across
    ``N_REPLICATES`` seeds. Cell ids are ``{sweep}-{value}-s{seed}``, so both the
    sweep and the condition are recoverable from the id alone and ``pytask -k
    confound-radial`` selects exactly one curve.

    Returns:
        Dict mapping cell id to the keyword arguments that produce it. The
        arguments are the complete definition of a cell -- the generator is
        deterministic given them -- so this dict is the whole design.
    """

    def sweep(name, argument, values):
        return {f"{name}-{value}": {argument: value} for value in values}

    conditions = {"base": {}}
    conditions |= sweep("separation", "separation", (0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0))
    conditions |= {
        f"continuum-{value}": {"structure": "continuum", "separation": value}
        for value in (2.0, 3.0, 6.0)
    }
    conditions |= sweep("anisotropy", "anisotropy", (0.3, 0.6, 0.9))
    # The same cones with the structure taken away: any k the sweep believes here
    # is manufactured by the cone alone.
    conditions |= {
        f"cone-only-{value}": {"separation": 0.0, "anisotropy": value}
        for value in (0.3, 0.6, 0.9)
    }
    conditions |= sweep("n-dims", "n_dims", (28, 256, 768))
    conditions |= sweep("intrinsic-dim", "intrinsic_dim", (4, 8, 32, 64))
    conditions |= sweep("noise", "noise", (0.25, 0.5, 1.0, 2.0))
    conditions |= {
        f"confound-{orientation}-{strength}": {
            "confound_orientation": orientation,
            "confound_strength": strength,
        }
        for orientation in ("axis", "radial")
        for strength in (0.5, 1.0, 2.0, 4.0, 8.0)
    }
    conditions |= {
        f"coupling-{value}": {
            "confound_orientation": "axis",
            "confound_strength": 4.0,
            "confound_cluster_coupling": value,
        }
        for value in (1.0, 2.0, 4.0)
    }
    conditions |= {
        f"imbalance-{name}": {"cluster_proportions": proportions}
        for name, proportions in {
            "graded": (0.4, 0.3, 0.2, 0.1),
            "dominant": (0.7, 0.1, 0.1, 0.1),
            "rare": (0.9, 0.05, 0.03, 0.02),
        }.items()
    }
    conditions |= sweep("n-patients", "n_patients", (5_000, 10_000))

    return {
        f"{name}-s{seed}": BASE | condition | {"random_state": RANDOM_STATE + seed}
        for name, condition in conditions.items()
        for seed in range(N_REPLICATES)
    }


def sweep_of(cell_id):
    """The sweep a cell belongs to, recovered from its id.

    ``{sweep}-{value}-s{seed}`` -- strip the seed, then the value. ``base`` has no
    value token and comes back as itself.
    """
    return cell_id.rsplit("-s", 1)[0].rsplit("-", 1)[0]


# The honest baseline and the primary arm. Both, for every cell.
SCALINGS = ("raw", "spherical")

# Sweeps where D1 is the instrument under test rather than a passenger: the
# separation curve it is calibrated on, the continuum it must not be fooled by,
# and the cone that -- per the Corpas trap -- is the classic way to fool it.
GATE_SWEEPS = (
    "base",
    "separation",
    "continuum",
    "anisotropy",
    "cone-only",
    "noise",
    "n-patients",
    "imbalance",
)

# Sweeps where the *difference between arms* is the question, so the gate runs on
# both. Elsewhere it runs on the primary arm only.
GATE_BOTH_ARMS = ("anisotropy", "cone-only")

PRIMARY_ARM = "spherical"


def runs_null_gate(cell_id, scaling):
    """Whether this cell-arm is in the declared gate subset.

    Args:
        cell_id: A key of ``GRID``.
        scaling: One of ``SCALINGS``.

    Returns:
        True if D1 should run for this pairing.
    """
    sweep = sweep_of(cell_id)
    if sweep not in GATE_SWEEPS:
        return False
    return scaling == PRIMARY_ARM or sweep in GATE_BOTH_ARMS


GRID = calibration_grid()
