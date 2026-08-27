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

Separation 2.0 is where detection begins, and the seeds spread widely there — margins of
0.212, 0.284 and 0.314 against an ARI of only 0.282. All three clear the threshold, but
the factor-of-1.5 spread across nothing but the seed is the honest width of the boundary
at this cohort size, and it is what sets the threshold: the band between 0.038 at
separation 1.5 and 0.212 here is empty, so 0.10 separates the two without sitting on top
of either.

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
filament — no partition to find — are called DISCRETE unanimously from separation 3
upward, at margins of 0.27 and then 0.78. A sliced continuum passes cluster tendency,
prediction strength and the null gate together.

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

### Tier 1: the evaluation protocol, and what the cohort permits

#### The probe reproduces the published benchmark

```{include} tables/tier1_oracle.md
```

Scored on the CLMBR features EHRSHOT itself released, our probe recovers the published
AUROC on all fourteen tasks, thirteen of them informatively. The largest tasks are the
sharpest test: `lab_anemia` reproduces to 0.0002 on 58,155 test rows and
`lab_thrombocytopenia` to 0.0004 on 56,338, where the confidence interval is narrow
enough that a protocol error of any consequence would show. The evaluation is therefore
the one behind the published numbers, and supervised results reported later rest on a
head that has been checked rather than assumed.

The single uninformative task is worth stating plainly, because it is the case that
motivated scoring against intervals rather than against a fixed tolerance. `new_celiac`
carries 94 positives in 7,129 rows; its interval spans 0.28 and contains both the
published value and chance. It selected a regularisation strength three orders of
magnitude from every other task, which is the same absence of signal seen from the other
side — with nothing to fit, the validation search has nothing to choose on. The task is
consistent with the published result and certifies nothing, and those are different
statements.

This establishes the evaluation and nothing upstream of it. Tokenisation and pooling are
untested here by construction, and are checked separately against the same released
vectors.

#### Truncation is the normal case, which is what makes context length measurable

```{include} tables/tier1_context.md
```

The median patient carries 3,129 clinical events, the mean 8,055, and the distribution
is extremely heavy-tailed — a twentieth of patients have fewer than 65 events and a
twentieth more than 31,854, with a maximum above 237,000. Records span a median of 8.9
years under observation.

The consequence is that no context length in the grid sees most patients whole. The
shortest holds the complete record for one patient in five and shows the median patient
about a sixth of theirs; the longest holds six in seven. **This is a precondition for
the study rather than a limitation of it.** Had most records fitted inside the shortest
window, the context-length contrast would have been null by construction and a quarter
of the model grid would have been measuring nothing that varies. The factor instead has
room across its entire range.

It also settles a design question that could otherwise have been decided on convenience.
Restricting analyses to patients whose histories fit every model's window would retain
the fifth of the cohort with the shortest records — precisely the patients for whom a
longer context cannot help — and would build the null result into the sample. Length is
therefore stratified and reported, never used to restrict.

Events approximate tokens without equalling them, so these shares are indicative until
recomputed through the tokeniser itself.

### What these results do not establish

The Tier 1 results concern the evaluation apparatus and the cohort, not any
representation. No model has been extracted and no comparison between spaces is reported
here; the oracle certifies the head that later comparisons will use, and the cohort
figures say what those comparisons will have room to detect.

The Tier 0 results carry a different limit. These are Gaussian spaces with orthonormally
embedded latent structure, and real embeddings are neither. They characterise what each
check can and cannot see under conditions we constructed; they do not estimate how often
such conditions arise in EHR foundation-model embeddings, and they provide no mapping
from a measured value on real data back to a latent structure. Their use is negative and
licensing: where a check is shown to be blind, a clean reading from it on real data
licenses nothing.
