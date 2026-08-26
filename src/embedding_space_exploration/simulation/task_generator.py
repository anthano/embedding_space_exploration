"""Materialise the Tier 0 calibration grid: one synthetic space per cell.

This file *is* the Tier 0 study design. ``generator.py`` can build any space; the
grid below declares which ones we actually evaluate, and every sweep exists to
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
  downstream of it.
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
"""

import pandas as pd
import pytask

from embedding_space_exploration.config import RANDOM_STATE, SIMULATION_DIR
from embedding_space_exploration.simulation.generator import simulate_embeddings

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


GRID = calibration_grid()

SPEC_FILES = {cell_id: SIMULATION_DIR / cell_id / "spec.parquet" for cell_id in GRID}


for cell_id, cell_kwargs in GRID.items():
    cell_produces = {
        name: SIMULATION_DIR / cell_id / f"{name}.parquet" for name in FRAMES
    }

    @pytask.task(id=cell_id, kwargs={"cell": cell_kwargs, "produces": cell_produces})
    def task_simulate_space(cell, produces):
        """Draw one synthetic space and write its five frames."""
        space = simulate_embeddings(**cell)
        for name, path in produces.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            space[name].to_parquet(path)


@pytask.task(kwargs={"specs": SPEC_FILES, "produces": SIMULATION_DIR / "cells.parquet"})
def task_collect_calibration_grid(specs, produces):
    index = pd.concat(
        [
            pd.read_parquet(path).assign(cell_id=cell_id)
            for cell_id, path in specs.items()
        ]
    )
    index.insert(0, "cell_id", index.pop("cell_id"))
    index.sort_values("cell_id").reset_index(drop=True).to_parquet(produces)
