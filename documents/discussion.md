## Discussion

:::\{note} Scaffold. Only the Tier 0 consequences are drafted; the rest follows the
argument once the empirical tiers are in. :::

### What the calibration changes about the battery

Three findings bear directly on how the battery should be read and reported.

**RankMe should not be reported as an intrinsic-dimension proxy.** It is not monotonic
in the planted dimensionality and is dominated by ambient noise. It may still be usable
as a within-model comparison under matched width and noise, but that is a much weaker
claim than the one its name invites, and the burden is now on anyone reporting it to say
which claim they mean.

**Confound orientation and confound decodability must both be reported.**
Principal-component regression misses a nuisance that a probe recovers almost perfectly
— not only in the constructed radial case, but for an ordinary linear confound whenever
it is not the dominant direction. Since both kinds destroy cluster recovery, orientation
alone does not license a claim that a space is clean.

**A cluster-tendency verdict cannot distinguish a weak partition from a continuum.**
Their margins overlap, so this is not a threshold that needs tuning. The discriminator
that did work — instability of the arg-max k across resampling seeds — is not currently
part of the battery, and adding it is the concrete proposal that follows from Tier 0.

### What this cannot settle

TODO. Extend the standing section from the battery document as the later domains land.
