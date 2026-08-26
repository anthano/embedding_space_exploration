## Introduction

:::\{note} Scaffold. The argument is developed in the project notes and is not yet
written up; the headings below fix the structure so later sections have something to
attach to. :::

### The reuse problem

TODO. Frozen EHR foundation-model embeddings are increasingly reused for unsupervised
work — subtyping, stratification, heterogeneity discovery — on the strength of a
validation record built entirely from supervised benchmarks. The claim of this paper is
that the transfer of trust is unearned.

### Why prediction validation does not transfer

TODO. Linear decodability is not metric structure: a probe needs a separating hyperplane
to exist, whereas clustering needs distances to mean something, and next-token
prediction constrains neither.

### How EHR sharpens the problem

TODO. The specific threat is confound-by-data-artifact — the leading axis of variation
being healthcare utilisation and documentation intensity rather than phenotype —
together with site effects and context-window truncation.

### A field that already ran this experiment

TODO. The single-cell literature, including the honest coda that its own battery
required validating {cite}`Luecken2022`.
