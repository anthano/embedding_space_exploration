"""All the general configuration of the project."""

from pathlib import Path

SRC: Path = Path(__file__).parent.resolve()
ROOT: Path = SRC.joinpath("..", "..").resolve()

BLD: Path = ROOT.joinpath("bld").resolve()

DOCUMENTS: Path = ROOT.joinpath("documents").resolve()

# Per-representation artifacts (embeddings, clusters, ...) under `MODELS / {key}`.
MODELS_DIR: Path = BLD.joinpath("models").resolve()

# Tier 0 calibration cells, one directory of frames per synthetic space, under
# `SIMULATION_DIR / {cell_id}`.
SIMULATION_DIR: Path = BLD.joinpath("simulation").resolve()

# Battery measurements scored against the Tier 0 ground truth, under
# `CALIBRATION_DIR / {cell_id} / {scaling}`.
CALIBRATION_DIR: Path = BLD.joinpath("calibration").resolve()


# ======================================================================================
# FROZEN ANALYSIS CONSTANTS
# ======================================================================================
# Decision rules from Study Design Freeze section 9 that more than one module reads.
# They are declared rather than left as defaults because a threshold chosen after seeing
# a sweep silently selects the answer -- and with the lockbox retired (see `splits`),
# these constants plus the selection-budget ledger are the only remaining guard against
# selection leakage. Changing one is a dated ledger entry, not an edit.
#
# The section 9 rules read by exactly one module live with that module, not here:
# `N_COMPONENTS` / `N_DROP_COMPONENTS` in `battery.prep`, `N_NULL_DRAWS` /
# `NULL_UPPER_PERCENTILE` in `battery.cluster_tendency`, `N_BOOTSTRAP_PARTITION` in
# `battery.stability`. The metric bootstrap (1,000 resamples) arrives with the module
# that performs it.

# Candidate cluster counts to sweep. The reference papers landed on 5 (Lian) and 7
# (Fan); 2-10 brackets that comfortably. Read by the sweep, the null gate and the
# partition bootstrap, which must all sweep the same k.
K_VALUES: tuple[int, ...] = tuple(range(2, 11))

# Largest k with mean prediction strength above this is selected. Lian used a strict
# 0.95 on >100k patients; 0.8 is the common default and more realistic at EHRSHOT's n.
PREDICTION_STRENGTH_THRESHOLD: float = 0.8

# Train/test resamples averaged per k for prediction strength. More repeats = a
# smoother curve; 20 is a reasonable cost at this n. The null gate must run the
# *identical* sweep for its band to be comparable, which is why this is shared.
N_REPEATS: int = 20

# The primary preprocessing geometry: L2 -> PCA(N_COMPONENTS) -> re-L2. "raw" and
# "standard" are sensitivity arms.
PRIMARY_SCALING: str = "spherical"

# PROPOSED, and Tier 0 is the evidence that should set it. Minimum share of the
# headroom above the covariance-matched null (see `battery.cluster_tendency`) for
# discrete structure to be believed. The verdict is *derived* from the recorded
# margin, so re-reading a finished run under a different value costs nothing --
# which is the point: the number can be fixed once the calibration sweep shows
# what a planted structure scores and what a continuum scores.
NULL_MARGIN_THRESHOLD: float = 0.25

# PROPOSED. The single k every space is clustered at when spaces are *compared*.
# Fixed rather than chosen per space: internal metrics are strongly k-dependent,
# so a space clustered at k=4 and one at k=7 have silhouettes that are not
# comparable, and the 18-way comparison silently stops being a comparison. The
# gate reports a margin instead of selecting k (`cluster_tendency`), so nothing
# in the label-free battery depends on this; it binds where an actual partition
# is needed -- Y2's clustering score above all. Value still open: the reference
# papers landed on 5 (Lian) and 7 (Fan), and Tier 0 should say whether the choice
# matters much within that range.
COMPARISON_K: int = 5

# One seed for everything that resamples, fits or permutes.
RANDOM_STATE: int = 0
