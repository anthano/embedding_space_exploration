## Results

### Tier 0: what the label-free battery detects

All results in this section are from synthetic Gaussian spaces with planted structure,
three replicate seeds per condition, read on the primary spherical preprocessing arm
unless stated. They characterise the *checks*, not any dataset.

#### Cluster detection has a sharp threshold, and the margin tracks recoverable structure

```{include} tables/tier0_separation.md
```

The gate is silent where it should be: at zero separation the margin is 0.002 and every
seed returns CONTINUOUS. Detection turns on sharply between separation 1.5 and 2.0, and
above that the margin tracks how much structure is actually recoverable almost
one-for-one — 0.65 against ARI 0.61, 0.86 against 0.85, 0.98 against 0.99. A margin is
therefore readable as an effect size and not only as a test statistic.

Separation 2.0 is the boundary, and the seeds disagree there: two return DISCRETE at
margins 0.284 and 0.314, one returns WEAK at 0.212, straddling the provisional
threshold. The boundary is real but not sharp at this cohort size.

Two checks come off badly. **Silhouette does not discriminate at all** in the regime
where detection is decided: it reads 0.049 on pure noise and 0.058 where structure is
detected at separation 2.0, rising only once the structure is already unmissable. And
the pipeline's own k selector returns k=2 at separation 2.0 on every seed, where the
margin's arg-max recovers the true k=4 on every seed — the margin locates the structure
that prediction-strength thresholding misses.

#### A continuum clears the gate, and no margin threshold separates it from weak clusters

```{include} tables/tier0_continuum.md
```

This is the failure the design most needed to check for. Points spread uniformly along a
filament — no partition to find — are called DISCRETE from separation 3 upward,
unanimously across seeds at separation 6 with margins of 0.77 to 0.79. A sliced
continuum passes cluster tendency, prediction strength and the null gate together.

The margins make the problem precise rather than solving it. Genuine blobs at separation
2 score 0.212–0.314 across seeds; a continuum at separation 3 scores 0.224–0.336. **The
ranges overlap**, so no threshold on the margin can separate weak-but-real clusters from
a continuum. Reporting an effect size instead of a verdict does not rescue the
distinction.

What does carry information is the stability of the arg-max k across seeds. Blobs return
k=4 on all three seeds at every separation from 2 upward. The continuum returns 2, 2, 2
at separation 2, then 2, 3, 4 at separation 3, and 3, 4, 4 at separation 6 — it never
settles. Seed-to-seed instability of the arg-max k is a continuum signature that costs
nothing to compute, and no check in the battery currently asks for it.

#### Confound orientation and confound decodability come apart

```{include} tables/tier0_confound.md
```

The two readings dissociate exactly as designed, and across the whole loading range
rather than at one extreme. A radially written confound registers 0.000 on the leading
principal component and 0.001 on variance-weighted association at every strength, while
a nonlinear probe recovers it at R² rising from 0.02 to 0.96. Orientation and
decodability are different properties of a space, and a battery reporting only the first
would call such a space clean.

The gap is not confined to the exotic construction. At low loading the **`axis`**
confound — a plain linear nuisance along a single direction — reads 0.003 on the leading
component while a linear probe recovers it at R² 0.963. Principal-component regression
only sees it once it dominates the spectrum, between strengths 1 and 2, where the
leading-component R² jumps from 0.008 to 0.950.

Both kinds destroy cluster recovery as they strengthen: ARI against the planted
partition falls from 0.61 to 0.02 for the axis confound and from 0.61 to 0.00 for the
radial one. A nuisance that C1 reads at 0.001 wrecks a partition as thoroughly as one it
reads at 0.95. This cuts against the convenient division of labour in which orientation
is what corrupts clustering and decodability is only what a probe exploits. Covariates
loading on nothing stayed between −0.10 and −0.13 throughout, so the positive readings
clear their floor comfortably.

#### RankMe does not track intrinsic dimension

```{include} tables/tier0_rankme.md
```

RankMe {cite}`Garrido2023` is not monotonic in the quantity it is being used to proxy.
Holding ambient noise fixed and varying the planted intrinsic dimension from 4 to 64,
RankMe *falls* from 79.9 to 65.9 and then rises to 85.5 — a range of about 20 across a
sixteen-fold change in the truth. Holding intrinsic dimension fixed at 16 and raising
ambient noise instead, it nearly doubles, from 65.9 to 126.8.

In this regime RankMe reports how much isotropic noise fills the ambient space, not how
many dimensions the structure occupies. It was calibrated on vision models and has not
been validated for this use; on EHR-shaped data it does not support an inference about
intrinsic dimensionality, and comparisons of RankMe across models of different width or
noise level are not comparisons of representational richness.

#### Anisotropy alone does not manufacture structure

```{include} tables/tier0_cone.md
```

A standing worry about cluster-tendency testing is that a single anisotropic blob will
produce an apparently good k≥2. Here it does not. Unstructured clouds pushed into cones
of increasing strength return CONTINUOUS on both preprocessing arms at every level, with
margins at or below zero even where 90% of the variance points one way. The gate is
robust to this failure mode, which is a point in favour of the covariance-matched null
the battery otherwise describes as weak.

The anisotropy knob is also recovered accurately, with measured mean cosine within 0.018
of the declared value across the range, so Domain A's anisotropy statistic reports what
it claims to.

### What Tier 0 does not establish

These are Gaussian spaces with orthonormally embedded latent structure, and real
embeddings are neither. The results characterise what each check can and cannot see
under conditions we constructed; they do not estimate how often such conditions arise in
EHR foundation-model embeddings, and they provide no mapping from a measured value on
real data back to a latent structure. Their use is negative and licensing: where a check
is shown to be blind, a clean reading from it on real data licenses nothing.
