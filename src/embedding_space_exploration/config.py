"""All the general configuration of the project."""

from pathlib import Path

SRC: Path = Path(__file__).parent.resolve()
ROOT: Path = SRC.joinpath("..", "..").resolve()

BLD: Path = ROOT.joinpath("bld").resolve()

DOCUMENTS: Path = ROOT.joinpath("documents").resolve()

# Per-representation artifacts (embeddings, clusters, ...) under `MODELS / {key}`.
MODELS_DIR: Path = BLD.joinpath("models").resolve()


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

# One seed for everything that resamples, fits or permutes.
RANDOM_STATE: int = 0
