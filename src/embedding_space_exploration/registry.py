"""The Tier 1 design: which representations to build, and where their artifacts go.

The counterpart to ``simulation.grid`` for real data, and kept apart from the task
files for the same reason -- ``@pytask.task`` registers into a global registry on
import, so a module that imports a task module to read a constant silently adds
that whole DAG to every later ``pytask.build()``. The design is data; only the
wiring is a task.

Deliberately *not* inherited from ``allofus.config``: that registry is two cells
(``gpt512``, ``llama2048``) with flat slugs for keys, which works when the only
question is "did the foundation model earn its keep" and breaks the moment the
factors themselves are the analysis. Here the id has to carry the factor levels,
because P1-P4 are queries over them.


The rule that decides cell from arm
===================================

**An axis that changes the embedding matrix is a cell. An axis that changes how
an existing matrix is read is an arm.**

That is the whole answer to "18 cells x 3 datasets x 4 preprocessing arms is a
different parametrization design, not a bigger tuple". Crossing them blindly is
what turns 28 extractions into 336 and buries the ~10 numbers the paper reports.

- **Cells** (produce a matrix): family, size, context length, pooling. Each owns
  one ``embeddings.parquet`` and is treated by everything downstream as just
  another space.
- **Arms** (re-read a matrix): preprocessing (``SCALINGS``), subsample n, k,
  cosine-vs-Euclidean, seed. All re-read one stored ``embeddings.parquet``.
- **Directory levels** (dataset, anchor): axes that must never be silently
  compared across. See below.

**Cells are not 1:1 with forward passes, and pooling is why.** Both readouts come
out of the same tensor: ``allofus`` already runs the model with
``output_hidden_states=True`` and takes ``hidden_states[-1][0, -1, :]``, so
last-token is an *index* into a ``[1, T, d]`` array that also contains every
position mean-pool would average. One pass per ``(dataset, anchor, family, size,
context)`` fans out to every pooling of it, at the cost of a second reduction and
a second parquet. So the extraction DAG is keyed on ``extraction_key()``, not on
``CELLS``, and pooling is a cell for what it *is* -- a distinct matrix, with its
own battery tree and its own row in the ledger -- not for what it costs.

Which means the P4 subset is a reporting decision, never an extraction one. Emit
mean-pool for every decoder cell because it is nearly free, then declare which
contrasts are primary. Restricting *extraction* to what you plan to report looks
like discipline and is really just a guarantee that checking a robustness
question later costs another GPU run.

Worth stating because it is the substance of P4: mean-pooling a **causal** model
averages hidden states over prefixes of every length, and position 0 has seen one
token. That is not a defect to be corrected -- it is the reason last-token is
standard for decoders, and the reason the comparison is interesting. The
bidirectional encoder has no such asymmetry, which is why ``ENCODER`` carries no
``last`` variant.


Why dataset and anchor are directories, not id tokens
=====================================================

Both change the matrix, so by the rule above they are cell-level -- but they are
also the two axes where an accidental cross-comparison is silently wrong rather
than merely noisy, so they are lifted into the path instead of the id.

``anchor`` especially. Study Design Freeze section 13 (2026-08-27) splits Y1 in
two: the **oracle** at every label's own ``prediction_time``, which is what makes
the published EHRSHOT AUROCs a correctness check, and the **experiment** at one
shared per-patient anchor, which is what makes Y1 and Y2 comparable to each
other. Those are different vectors for the same patient and the same model, and
the headline claim -- a ranking flip between the two dependent variables -- is
only attributable to the task if they never mix. A directory level makes mixing
them a missing-file error rather than a wrong number.


Ids
===

``{family}-{size}-{context}-{pooling}`` for models, dash-joined over the fields a
representation actually has, so baselines are just ``tfidf`` / ``clinical`` /
``nuisance``. Pooling is always explicit, never defaulted: it is a factor in
Study Design Freeze section 4, and a factor that only appears in the id when it
takes its non-default level is a factor you will forget to report.

Ids are parsed by lookup (``CELLS[cell_id]``), never by splitting -- the
dataclass is the source of the factor levels and string surgery on ids is how a
grid quietly disagrees with itself. ``grid.sweep_of`` gets away with splitting
because a Tier 0 cell has no attributes beyond its id; a Tier 1 cell has nine.

The ids stay long and greppable rather than short and dense, because
``pytask -k`` selection is the property that made the Tier 0 scheme usable:
``-k 4096`` is the context sweep, ``-k mamba`` is one family, ``-k mean`` is P4,
``-k mimic`` is the site-shift arm.
"""

from dataclasses import asdict, dataclass, fields

from embedding_space_exploration.config import BLD

# ======================================================================================
# Directory-level axes
# ======================================================================================

# Tier 1 is EHRSHOT alone. MIMIC is Tier 3b and gated on B2; AoU is Tier 3 and
# runs a costed subset of CELLS rather than all of it (see All Of Us Computing
# Costs) -- which is a reason to keep the dataset out of the cell id, since the
# cell set is not the same in every dataset.
DATASETS = ("ehrshot", "mimic", "aou")

# Where each patient's history is cut. Never compared across; see module docstring.
#   "perlabel"  -- every label's own prediction_time. The Wornow2025 protocol,
#                  and the only anchor whose numbers are comparable to published
#                  ones. Correctness oracle only.
#   "shared"    -- one per-patient anchor, the last event before the declared
#                  outcome window. The experiment. Carries both Y1 and Y2.
#   "lastevent" -- the patient's final event, ignoring any window. Robustness arm
#                  for "shared", costed at ~2% of Y1.
ANCHORS = ("perlabel", "shared", "lastevent")

# Preprocessing arms. "raw" and "spherical" are `prepare_matrix` scalings; the
# drop2 and residual arms are the same spherical scaling at a different
# `n_drop_components` / residualisation setting, named apart because Study Design
# Freeze section 9 requires corrections to be *measured arms, never fixes* -- an
# arm that shares a directory with the thing it corrects cannot be read as one.
SCALINGS = ("raw", "spherical", "spherical-drop2", "spherical-resid", "standard")

PRIMARY_ANCHOR = "shared"
PRIMARY_SCALING_ARM = "spherical"


# ======================================================================================
# Cells
# ======================================================================================


@dataclass(frozen=True)
class Cell:
    """One representation: everything needed to produce one embedding matrix.

    Attributes:
        family: ``gpt`` / ``llama`` / ``mamba`` / ``hyena`` / ``motor`` /
            ``modernbert``, or the baseline's own name.
        size: Parameter tier as the model card names it, or ``None`` for
            baselines. Note this is *not* free to vary against ``family``: the
            Context Clues collection ships GPT and Llama at ``base`` and Mamba
            and Hyena at ``tiny`` / ``large``, so architecture and parameter
            count move together across those pairs. Recorded here so that
            confound is visible in the frame rather than remembered.
        context: Trained context window. **Not comparable across ``token_unit``**
            -- see that field.
        pooling: ``last`` for causal decoders (the only position that has seen
            the whole sequence), ``mean`` for the bidirectional encoder (where
            every position has, and mean is the standard readout), ``None`` for
            baselines. The cross of pooling with architecture is P4; the
            encoder+``last`` cell is deliberately absent because it is vacuous.
            Cells differing only in this field share one forward pass -- see
            ``extraction_key``.
        objective: ``nexttoken`` / ``timetoevent`` / ``mlm`` / ``none``. The P3
            contrast, and the mechanism claim in Paper Outline section 2.
        token_unit: ``code`` or ``text``. The encoder arm eats serialised text,
            where one OMOP code becomes several word-pieces, so its 8192 covers
            far fewer *events* than a decoder's 8192. Any figure with context on
            an axis must either facet on this field or use the measured
            events-covered from ``extraction.json`` instead.
        loader: Which extraction path builds it -- ``hf_ehr`` (Context Clues),
            ``motor`` (FEMR), ``text`` (serialise then encode), ``baseline``.
        source: HuggingFace repo id, or ``None`` for baselines.
    """

    family: str
    size: str | None
    context: int | None
    pooling: str | None
    objective: str
    token_unit: str
    loader: str
    source: str | None

    @property
    def id(self):
        """Dash-joined over the fields the representation actually has."""
        parts = (self.family, self.size, self.context, self.pooling)
        return "-".join(str(part) for part in parts if part is not None)


def _clmbr(family, size, context, pooling="last"):
    """A Context Clues cell. All 16 share CLMBRTokenizer and the Stanford corpus."""
    return Cell(
        family=family,
        size=size,
        context=context,
        pooling=pooling,
        objective="nexttoken",
        token_unit="code",
        loader="hf_ehr",
        source=f"StanfordShahLab/{family}-{size}-{context}-clmbr",
    )


# The spine: context length with architecture *and* parameter count held fixed,
# four independent replications across a 32x range. Wornow2025 showed longer
# context improves prediction; if it monotonically degrades geometry across all
# four families, that is the section 2 dissociation measured directly.
CONTEXT_CLUES = (
    *(_clmbr("gpt", "base", ctx) for ctx in (512, 1024, 2048, 4096)),
    *(_clmbr("llama", "base", ctx) for ctx in (512, 1024, 2048, 4096)),
    *(_clmbr("mamba", "tiny", ctx) for ctx in (1024, 4096, 8192, 16384)),
    *(_clmbr("hyena", "large", ctx) for ctx in (1024, 4096, 8192, 16384)),
)

# P3, the objective contrast, and as close to a controlled experiment as the
# field hands us: same institution, same corpus, same hidden dim, near-identical
# window, ~20% parameter difference. Only the objective moves.
MOTOR = Cell(
    family="motor",
    size="base",
    context=496,
    pooling="last",
    objective="timetoevent",
    token_unit="code",
    loader="motor",
    source="StanfordShahLab/motor-t-base",
)

# The MLM arm, and the only one obtainable: verified 2026-08-27 that no
# structured-EHR BERT has usable public weights (Med-BERT refused contractually,
# CEHR-BERT / CORE-BEHRT / reAIM-Lab code-only, MedRep releases concept
# representations but not its backbones). So the objective contrast has to be
# reached by serialising to text -- which is the Hegselmann2026 recipe, minus its
# one removable defect: their best encoder config averaged up to 16 separately
# encoded 512-token chunks, so the record was never seen whole. At 8192 this is
# one forward pass over their identical input. See the Build Plan encoder-arm
# item; mean-pooling is the *correct* readout here and is not the thing 8192 fixes.
ENCODER = Cell(
    family="modernbert",
    size="base",
    context=8192,
    pooling="mean",
    objective="mlm",
    token_unit="text",
    loader="text",
    source="thomas-sounack/BioClinical-ModernBERT-base",
)

# P4, on every decoder rather than a declared subset. An earlier draft restricted
# this to seven cells on the belief that each cost its own forward pass; it does
# not (see the module docstring), so the restriction bought nothing and would have
# made every later robustness question a new GPU run. Which contrasts are
# *primary* is still declared in advance -- that is the part the selection budget
# cares about, and it belongs in the analysis, not the extraction.
#
# The encoder is absent by design: bidirectional, so no last-token counterpart.
MEAN_POOL_CELLS = tuple(
    Cell(**{**vars(cell), "pooling": "mean"}) for cell in (*CONTEXT_CLUES, MOTOR)
)


def _baseline(name, loader="baseline"):
    return Cell(
        family=name,
        size=None,
        context=None,
        pooling=None,
        objective="none",
        token_unit="code",
        loader=loader,
        source=None,
    )


# Controls, not courtesies (Paper Outline section 6). "nuisance" is the sharpest:
# five dimensions of pure utilisation, and given the AoU C3 result (log n_events
# eta^2 = 0.568) it may score uncomfortably well -- which is the cleanest single
# test of the "fitting meaningless noise" critique.
BASELINES = (_baseline("tfidf"), _baseline("clinical"), _baseline("nuisance"))

# The 18 that Study Design Freeze section 3 calls "~18 frozen cells": the spine,
# the objective contrast, the encoder arm. Everything else is a control or a
# within-model arm, and keeping the count derived rather than asserted is what
# the selection-budget ledger needs.
MODEL_CELLS = (*CONTEXT_CLUES, MOTOR, ENCODER)

CELLS = {cell.id: cell for cell in (*MODEL_CELLS, *MEAN_POOL_CELLS, *BASELINES)}


# ======================================================================================
# Selection-budget ledger support
# ======================================================================================
# Study Design Freeze section 13 wants one auditable row per configuration
# actually evaluated, "auditable rather than reconstructed at write-up". It is
# only auditable if it is *derived*: every frame the battery writes carries these
# columns, so the ledger is a concat over bld rather than a thing anyone
# maintains by hand. With the lockbox retired this and the frozen constants are
# the only guards left against selection leakage, so it is not bookkeeping.

ID_COLUMNS = (
    "dataset",
    "anchor",
    "cell_id",
    "scaling",
    *(f.name for f in fields(Cell)),
)


def id_columns(dataset, anchor, cell_id, scaling):
    """The leading identity columns for any frame written under Tier 1.

    Returns:
        Dict in ``ID_COLUMNS`` order, ready to prepend to a result frame.
    """
    return {
        "dataset": dataset,
        "anchor": anchor,
        "cell_id": cell_id,
        "scaling": scaling,
        **asdict(CELLS[cell_id]),
    }


# ======================================================================================
# Layout
# ======================================================================================
# bld/tier1/{dataset}/{anchor}/
#     cells/{cell_id}/embeddings.parquet     <- one forward pass, the expensive half
#     cells/{cell_id}/extraction.json        <- provenance: untruncated length, events
#                                               covered, model source, wall clock
#     battery/{cell_id}/{scaling}/*.parquet  <- cheap, re-reads the matrix above
#     compare/{scaling}/*.parquet            <- cross-cell: CKA, P1-P4, roll-ups
#
# `compare` sits beside the per-cell trees rather than inside one because CKA and
# the paired contrasts are not properties of any single cell, and a comparison
# filed under one of its own operands is a comparison someone will later read as
# a property of that operand.

TIER1_DIR = BLD.joinpath("tier1").resolve()


def extraction_key(cell):
    """What one forward pass is keyed on: everything about a cell except pooling.

    Cells sharing a key are different reductions of the same ``[1, T, d]`` tensor,
    so they cost one pass between them. ``None`` for baselines, which run no model.
    """
    if cell.source is None:
        return None
    return (cell.family, cell.size, cell.context)


def poolings_by_extraction(cell_ids=None):
    """Group cell ids by forward pass: ``{extraction_key: (cell_id, ...)}``.

    The extraction DAG iterates this rather than ``CELLS`` -- one task per group,
    emitting one ``embeddings.parquet`` per member.
    """
    grouped = {}
    for cell_id in CELLS if cell_ids is None else cell_ids:
        key = extraction_key(CELLS[cell_id])
        if key is not None:
            grouped.setdefault(key, []).append(cell_id)
    return {key: tuple(members) for key, members in grouped.items()}


def cell_dir(dataset, anchor, cell_id):
    """Where one cell's embedding matrix and its extraction provenance live."""
    return TIER1_DIR / dataset / anchor / "cells" / cell_id


def battery_dir(dataset, anchor, cell_id, scaling):
    """Where one cell-arm's battery measurements live."""
    return TIER1_DIR / dataset / anchor / "battery" / cell_id / scaling


def compare_dir(dataset, anchor, scaling):
    """Where cross-cell comparisons (CKA, P1-P4, roll-ups) live."""
    return TIER1_DIR / dataset / anchor / "compare" / scaling


def task_id(dataset, anchor, cell_id, scaling=None):
    """The pytask id for a cell or cell-arm.

    Long and greppable on purpose -- ``pytask -k`` selection over the factor
    levels is the property worth keeping (``-k 4096``, ``-k mamba``, ``-k mean``).
    """
    parts = (dataset, anchor, cell_id, scaling)
    return "-".join(part for part in parts if part is not None)
