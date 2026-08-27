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

# Set 2026-08-27 from the Tier 0 separation sweep (spherical arm, 3 seeds/cell).
# Minimum share of the headroom above the covariance-matched null (see
# `battery.cluster_tendency`) for discrete structure to be believed. 0.10 is the
# midpoint of the empty band the calibration left behind: planted clusters at
# separation 1.5 -- weak but real, ARI 0.14 -- top out at a margin of 0.038,
# separation 2.0 (ARI 0.28) starts at 0.212, and nothing lands in between.
# Separation 0 reads -0.008 to +0.003, so the null end yields no false positives.
#
# Two things this threshold is deliberately *not* asked to do.
#
# It does not separate a continuum from weak clusters. A continuum at separation
# 3 scores 0.224-0.336 against genuine blobs at separation 2 at 0.212-0.314; the
# ranges overlap, so no value of this constant can carry that distinction.
# Seed-to-seed instability of the gate's arg-max k is the D6 discriminator
# instead -- blobs return the same k on every seed, a continuum never settles.
#
# It is not comparable across n. At fixed planted structure (separation 3, ARI
# flat at 0.61-0.63) the margin climbs 0.65 -> 0.80 -> 0.87 as n goes
# 2,000 -> 5,000 -> 10,000, so the subsample arms must have their verdicts read
# against their own n rather than against this one number.
#
# The verdict is *derived* from the recorded margin, so re-reading a finished run
# under a different value costs nothing. Changing it is a dated ledger entry.
NULL_MARGIN_THRESHOLD: float = 0.10

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
