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

### Tier 1: EHRSHOT, and the two questions the oracle separates

Tier 1 runs on EHRSHOT {cite}`Wornow2023`, a longitudinal benchmark of 6,739 patients
drawn from one academic health system, with a native patient-level train/validation/test
partition and fourteen binary prediction tasks. The cohort is taken exactly as released.
A custom split would re-partition it and break the published-baseline comparison, which
is the cheapest correctness check available on the whole pipeline; the three roles a
held-out split is meant to serve are already carried by the benchmark's own — training
fits, validation selects, test is scored once.

The supervised probe follows the Context Clues protocol {cite}`Wornow2025` rather than a
protocol of our own: frozen representations, a logistic-regression head, the
regularisation strength tuned on the validation split, the test split scored once, and
every AUROC reported with a bootstrapped 95% confidence interval over 1,000 resamples of
that split. A comparison against published numbers only certifies anything if the
protocol producing our numbers is the protocol that produced theirs.

**The correctness check is two questions, and running them together answers neither.**
An embedding pipeline can fail at tokenisation, at pooling, or at evaluation, and a
single downstream AUROC compared against a paper cannot say which. EHRSHOT ships the
CLMBR embeddings it evaluated — one vector for each of the 406,379 distinct (patient,
prediction time) pairs across its benchmark — which allows the two to be separated.
Running our probe on *their* vectors isolates the evaluation protocol and tests nothing
upstream of it. Comparing *our* vectors against theirs at matched (patient, prediction
time) then isolates extraction, and does so directly rather than through an AUROC that
confounds both. Only the first is reported here; the second belongs with the extraction
it checks.

A task counts as reproduced when the published AUROC falls inside our confidence
interval, rather than within a fixed distance of our estimate. The benchmark's test
splits differ in size by two orders of magnitude, and so do the intervals — from 0.003
on the largest laboratory task to 0.28 on the rarest diagnosis — so no single absolute
tolerance is defensible at both ends. One that admits sampling noise on the rare task
would wave through a catastrophic failure on the common one. Tasks whose interval is too
wide to distinguish success from failure are marked as such rather than counted as
evidence.

### Anchoring, and why one anchor carries both outcomes

An embedding is taken at a point in a patient's timeline, and which point is a design
choice that decides what a comparison can mean. The claim this study is built to test is
that a space's ranking *changes with the downstream task* — that a representation which
predicts well may cluster poorly. Attributing such a reversal to the task requires
everything else to be held fixed: the same vectors, the same patients, the same label.
Reading supervised performance off one matrix and clustering off another would leave the
reversal with a second and entirely plausible explanation, since models with longer
context windows should fare relatively better wherever histories are longer.

Both outcomes are therefore read from a single per-patient representation, anchored at
each patient's last event before a declared outcome window, with the supervised label
and the clustering phenotype defined strictly inside that window. The window is a
temporal firewall. Without it the codes that define the phenotype sit in the history the
model encoded, clusters recover it tautologically, and the gap between what a probe
extracts and what a partition recovers — the object of study — stops being measurable.

The anchor resolves a genuine tension rather than dissolving it. The firewall argues for
anchoring early; the context-length contrast argues for anchoring late, because
truncating histories is exactly the manipulation that would make long context useless by
construction. The last event before the window is the latest anchor the firewall
permits.

This leaves two supervised evaluations with different jobs, and they are not
interchangeable. The oracle scores every label at its own prediction time, as the
published protocol requires, and its result is a statement about correctness. The
experiment scores the shared anchor, and its result is a ranking across representations.
The experiment's absolute values are not comparable to published ones and are not
intended to be; the oracle has already established that the pipeline is sound, and a
ranking does not need an external target.

### Frozen constants

The decision rules are declared before the run rather than chosen after it: a threshold
picked once a sweep has been seen silently selects the answer. Candidate cluster counts
are 2–10, the prediction-strength threshold is 0.8, PCA retains 50 components with no
leading components dropped, the null gate draws 20 covariance-matched datasets and
compares against their 95th percentile, the partition bootstrap is 25 resamples, and one
seed drives everything that resamples, fits or permutes.

The null-margin threshold separating a weak from a believed structure is the one
constant set from this tier rather than before it, at 0.10.
{numref}`tbl-tier0-separation` is the evidence: planted structure too weak to recover
tops out at a margin of 0.038, structure that is recoverable starts at 0.212, and no
condition lands between them, so the value is the midpoint of an empty band rather than
a cut through a distribution. Two limits are recorded with it. It does not separate a
continuum from weak clusters — {numref}`tbl-tier0-continuum` shows those ranges
overlapping, so no value could — and it is not comparable across cohort sizes, since at
fixed planted structure the margin climbs from 0.65 to 0.87 as n goes from 2,000 to
10,000 while the recoverable structure stays flat.
