## Methods

### Tier 0: calibrating the instrument before using it

Every check in a validation battery claims to detect something. Whether it does, and
where it stops doing so, is an empirical question about the check rather than about any
dataset — and it is cheaper to answer on synthetic data than to discover on real data
after a claim has been made. This is the lesson of the single-cell literature, where the
scIB battery {cite}`Luecken2022` became a standard before a later evaluation showed it
misses intra-cell-type conservation: batteries need validating too.

Tier 0 therefore builds embedding spaces whose structure is *planted* and known, runs
the battery against them, and records where each check fires, where it stays silent, and
where it is confidently wrong.

### Synthetic spaces with planted ground truth

Each synthetic space is built in a low-dimensional latent space and then embedded in a
wider ambient one, in three blocks that do not interfere: a **signal** block carrying
cluster structure, a **confound** block carrying a nuisance variable, and an isotropic
**background** block, together summing to the declared intrinsic dimension. The latent
is projected through an orthonormal basis, so the noiseless matrix has rank exactly the
intrinsic dimension and within-cluster scatter is preserved; isotropic ambient noise is
added in every ambient direction; and the whole cloud is finally offset along one shared
direction to produce anisotropy.

Four properties are controlled independently, matching the four things the battery
claims to measure:

- **Cluster structure.** `n_clusters` centres placed mutually equidistant about the
  origin, `separation` apart in units of within-cluster SD. The centres are centred on
  the origin deliberately: placed in the positive orthant they give the cloud a cone of
  its own, and separation would silently move the anisotropy that the anisotropy knob is
  supposed to own. A `continuum` mode spreads points uniformly along the polyline
  through the *same* centres — identical extent and local scatter, no discrete
  structure.
- **Confound loading.** A nuisance covariate written into the space in one of two
  orientations. `axis` writes it along a single direction, so principal-component
  analysis sees it as a component. `radial` writes it as a *radius within a plane at
  uniform angle*: the distance from the origin is a strictly monotone function of the
  covariate while the direction is independent of it, so every fixed direction has zero
  linear association with it in expectation, yet a nonlinear learner recovers it from
  the squared norm. A coupling parameter additionally entangles the confound with
  cluster identity.
- **Anisotropy.** Declared in the units the check reports it in — mean cosine to the
  centroid — by offsetting every row along a shared direction with norm
  `r · a / sqrt(1 - a²)` for mean row norm `r`. "Did the check recover it?" is then a
  subtraction.
- **Intrinsic dimension.** The exact rank of the noiseless matrix, sunk into a wider
  ambient space and blurred by ambient noise. Above zero noise the matrix is numerically
  full-rank, and only an *effective* rank measure can still report the planted value.

Ground truth is held in a separate frame and never passed to a check.

### The calibration grid

The grid moves one knob at a time from a reference cell of 2,000 patients, 128 ambient
dimensions, intrinsic dimension 16, and four clusters three within-cluster SDs apart. It
is deliberately not a full crossing: 46 conditions rather than the thousands a product
would give, so each sweep reads as one knob and one curve. Sweeps cover cluster
separation from 0 to 6; the continuum comparator; anisotropy, both with structure
present and with structure removed; ambient width from 28 to 768 and intrinsic dimension
from 4 to 64; ambient noise; confound loading from 0.5 to 8 in both orientations;
confound–cluster coupling; cluster size imbalance; and cohort size at 2,000, 5,000 and
10,000.

Every condition is drawn at three replicate seeds, because a calibration table has to
report a band per condition rather than a single draw. Every cell also carries
covariates that load on nothing, so each check's false-positive floor is measured
wherever the check is measured. The grid is 138 cells in total.

### Checks evaluated, and what is out of reach

Each cell is scored on both preprocessing arms — the untouched `raw` baseline and the
primary `spherical` arm (L2-normalise, PCA to 50 components fit on the train split,
re-L2-normalise) — because the arm sits between the space and every clustering check, so
a check's behaviour is a property of the pair.

Domain A (well-formedness, anisotropy, effective rank), Domain C1 (confound orientation
by principal-component regression), Domain D (cluster tendency against a
covariance-matched null, prediction strength, bootstrap stability, internal metrics) and
the confound-decodability probes are all in scope.

Domain B is **not**, and this is a limit of the design rather than an omission. Order
ablation, input faithfulness and kNN concordance take a patient timeline and a model,
not an embedding matrix; a synthetic space has no input to ablate. Vocabulary coverage
(A1) is out for the same reason. These checks are exercised only on real data, and
nothing in Tier 0 licenses any claim about them.

### Scoring

Checks are scored in two ways, and keeping them apart matters more than any individual
number.

Checks with a **truth counterpart** — cluster ARI against the planted labels, RankMe
against intrinsic dimension, mean cosine against declared anisotropy — yield a
measured-minus-declared error, and an error curve across a sweep is a calibration in the
ordinary sense.

Checks with **no counterpart** — prediction strength, the internal metrics, the
cluster-tendency verdict — have nothing to subtract from, so they are read as behaviour
along a sweep: at what separation does the verdict flip, and does it also flip on a
continuum, where flipping is wrong. Such a check cannot be scored pointwise, but it can
still be caught being confidently wrong.

### The cluster-tendency gate reports a margin, not a k

The null gate compares the real prediction-strength sweep against the same sweep
computed on draws from a single covariance-matched Gaussian {cite}`Dinga2019`. Its
original decision rule took the *largest* k that both beat the null band and cleared an
absolute prediction-strength threshold. That rule fails in a reproducible way.
Prediction strength decays with k for real and null data alike, so exceeding the null
becomes close to automatic in the high-k tail, and "largest" resolves to the weakest
evidence in the table; lowering the threshold to admit a genuine k admits several
spurious larger ones first.

The gate therefore reports the share of the **headroom above the null** that a space
captures,

$$ \text{margin} = \frac{\text{real} - \text{null median}}{1 - \text{null median}}, $$

and does not select k at all. Prediction strength is bounded by 1, so the margin is
bounded by 1 at every k, which the two obvious alternatives are not: the raw difference
favours low k, where the whole scale is larger, and the ratio favours high k, where its
ceiling is larger. The verdict is *derived* from the recorded margin, so re-reading a
finished run under a different threshold costs nothing.

Selecting k per space would in any case make a comparison across spaces incoherent,
since internal metrics are strongly k-dependent and a space clustered at k=4 and one
clustered at k=7 have silhouettes that cannot be compared. Where an actual partition is
required, k is fixed and declared across spaces.

### Frozen constants

The decision rules are declared before the run rather than chosen after it: a threshold
picked once a sweep has been seen silently selects the answer. Candidate cluster counts
are 2–10, the prediction-strength threshold is 0.8, PCA retains 50 components with no
leading components dropped, the null gate draws 20 covariance-matched datasets and
compares against their 95th percentile, the partition bootstrap is 25 resamples, and one
seed drives everything that resamples, fits or permutes. The null-margin threshold
separating a weak from a believed structure is provisional, and
{numref}`tbl-tier0-continuum` is the evidence bearing on it.
