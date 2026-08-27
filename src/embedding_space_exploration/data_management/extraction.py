"""Patient embeddings: one forward pass per model, every pooling read off it.

The expensive half of Tier 1, and the reason nothing downstream exists yet. Given
an index of ``(person_id, cutoff)`` anchors and a ``registry`` cell, this
produces the ``embeddings.parquet`` that Domains A-D, both dependent variables
and every cross-cell comparison then re-read.

Split the way ``timeline`` is split, and for the same reason. The pure half --
the two masked reductions, the truncation accounting -- operates on arrays,
needs neither ``torch`` nor ``hf_ehr`` nor a licensed extract, and is where the
mistakes that survive a green run live. The impure half loads a model and runs
it. Only the second half needs a GPU or a gated download, so only the second
half is hard to test, and it is deliberately thin.

Lives in ``data_management`` rather than ``analysis`` because it is the last
step that touches the licensed extract: everything above it reads matrices.


Truncation keeps the tail, and this is load-bearing
===================================================

``CLMBRTokenizer`` inherits HuggingFace's default ``truncation_side='right'``,
which keeps a record's *oldest* tokens and discards its most recent. For a
timeline cut at an anchor that is exactly backwards -- the embedding is supposed
to describe the patient at the anchor, and right-truncation hands the model the
patient's distant past instead, then reads a "last token" that is nowhere near
the anchor. Measured on ``gpt-base-512-clmbr`` with a 2,880-token timeline, the
two sides give last-token vectors at **cosine 0.52**: not a rounding difference,
a different patient.

It would also silently destroy P1, the context-length spine. Under
right-truncation every cell in a family reads the same opening 512 tokens from
the record's beginning, the extra context of the 1024/2048/4096 cells covers
history the patient already had, and "longer context does not change the
geometry" becomes an artifact of the tokeniser rather than a finding. A null
built into the design is the failure mode Study Design Freeze section 7 names
twice, so ``TRUNCATION_SIDE`` is set explicitly here and asserted in the tests
rather than left to a default that a ``transformers`` bump could flip.

The untruncated length is measured *before* the cut regardless, because it is
A3's input and the stratifier that section 7 requires: cap-hit rates run from
~62% at 512 tokens to near-zero at 16384, so metrics are reported within length
stratum and never on a restricted common-eligible set.


Pooling is masked, because padding is not history
=================================================

Both readouts come out of one ``[B, T, d]`` tensor -- that is why pooling is not
a second forward pass -- but both are wrong if they ignore the attention mask.
With right padding, an unmasked ``[:, -1, :]`` reads a PAD position for every
sequence in the batch shorter than the longest, and an unmasked mean divides a
real sum by a padded count. Batch size 1 hides this and the throughput this
anchor needs does not permit batch size 1: the ``perlabel`` anchor is ~406k
forward passes against the ``shared`` anchor's ~6.7k.
"""

import time

import numpy as np
import pandas as pd

from embedding_space_exploration.data_management.timeline import (
    events_until,
    patient_events,
    to_model_events,
)

# Keep the most recent tokens at the anchor. See the module docstring -- this is
# not a default, it is a finding.
TRUNCATION_SIDE = "left"

# Rows per forward pass. Sequences are padded to the longest member, so a batch
# of mixed lengths wastes compute on padding; the caller can lower this on a
# long-context cell where memory binds instead.
BATCH_SIZE = 8


# ======================================================================================
# The pure half: reductions and truncation accounting
# ======================================================================================


def pool_last(hidden, mask):
    """The last *real* position of each sequence.

    The only position a causal decoder has let see the whole history, which is
    what makes it the standard readout for one -- and what makes reading it off
    a PAD position a silent corruption rather than a crash.

    Args:
        hidden: ``[B, T, d]`` array of final-layer hidden states.
        mask: ``[B, T]`` attention mask, 1 for real tokens and 0 for padding.

    Returns:
        ``[B, d]`` array.
    """
    lengths = _lengths(mask)
    return hidden[np.arange(hidden.shape[0]), lengths - 1, :]


def pool_mean(hidden, mask):
    """The mean over each sequence's real positions.

    P4's arm. Worth restating that on a *causal* model this averages hidden
    states taken over prefixes of every length, and position 0 has seen one
    token -- not a defect to correct, but the reason last-token is standard for
    decoders and the reason the contrast is interesting at all.

    Args:
        hidden: ``[B, T, d]`` array of final-layer hidden states.
        mask: ``[B, T]`` attention mask, 1 for real tokens and 0 for padding.

    Returns:
        ``[B, d]`` array.
    """
    weights = mask[:, :, None].astype(hidden.dtype)
    return (hidden * weights).sum(axis=1) / _lengths(mask)[:, None]


POOLERS = {"last": pool_last, "mean": pool_mean}


def pool(hidden, mask, pooling):
    """Apply one named reduction.

    Args:
        hidden: ``[B, T, d]`` array of final-layer hidden states.
        mask: ``[B, T]`` attention mask.
        pooling: ``last`` or ``mean``.

    Returns:
        ``[B, d]`` array.

    Raises:
        KeyError: If ``pooling`` is not a known readout.
    """
    return POOLERS[pooling](hidden, mask)


def truncation_report(untruncated, context):
    """What the cut cost, per patient.

    A3's input and section 7's stratifier in one frame. ``covered`` is the share
    of the patient's tokens the model actually saw, which is the quantity to
    compare across ``token_unit`` -- a text cell's 8192 word-pieces and a code
    cell's 8192 codes cover wildly different amounts of record, so any figure
    with context on an axis reads this rather than the nominal window.

    Args:
        untruncated: Sequence of token counts before truncation.
        context: The cell's trained context window.

    Returns:
        DataFrame with ``n_tokens_untruncated``, ``n_tokens_seen``,
        ``truncated`` and ``covered``.
    """
    lengths = np.asarray(untruncated, dtype=int)
    seen = np.minimum(lengths, context)
    return pd.DataFrame(
        {
            "n_tokens_untruncated": lengths,
            "n_tokens_seen": seen,
            "truncated": lengths > context,
            "covered": np.divide(
                seen,
                lengths,
                out=np.ones(len(lengths), dtype=float),
                where=lengths > 0,
            ),
        }
    )


def _lengths(mask):
    """Real-token count per row, floored at 1 so an empty history cannot divide by 0."""
    return np.maximum(np.asarray(mask).sum(axis=1).astype(int), 1)


def _cutoff(value):
    """Normalise a frame's missing cutoff to the ``None`` ``events_until`` expects.

    ``events_until`` takes ``None`` to mean "keep everything", but a cutoff column
    read out of a DataFrame carries ``NaT`` instead, and every ``event_time <=
    NaT`` comparison is ``False`` -- so an unnormalised ``NaT`` silently empties
    the history rather than keeping it, and the patient is embedded as a single
    PAD token. Caught in the first smoke run over real patients, where it looked
    like five identical vectors and a token count of zero.
    """
    return None if value is None or pd.isna(value) else value


# ======================================================================================
# The impure half: load a model, run it
# ======================================================================================


def load_extractor(cell):
    """Load one cell's tokeniser and model, with truncation pinned to the tail.

    Imported lazily and kept to one function so the pure half above stays
    runnable with neither ``torch`` nor ``hf_ehr`` installed.

    Args:
        cell: A ``registry.Cell`` with ``loader == "hf_ehr"``.

    Returns:
        Tuple of ``(tokenizer, model)``, the model in eval mode.

    Raises:
        NotImplementedError: For the ``motor``, ``text`` and ``baseline``
            loaders, which are separate integrations and deliberately deferred
            until CKA says whether they are worth building.
    """
    if cell.loader != "hf_ehr":
        raise NotImplementedError(
            f"loader {cell.loader!r} is not built: the staged sequence in Build Plan "
            "section 1.2 stops after the Context Clues loader and lets CKA decide "
            "whether the MOTOR and encoder integrations happen at all"
        )
    from hf_ehr.data.tokenization import CLMBRTokenizer
    from transformers import AutoModelForCausalLM

    tokenizer = CLMBRTokenizer.from_pretrained(cell.source)
    tokenizer.truncation_side = TRUNCATION_SIDE
    model = AutoModelForCausalLM.from_pretrained(cell.source)
    model.eval()
    return tokenizer, model


def count_tokens(tokenizer, events):
    """Tokens this timeline would occupy before any truncation.

    Measured on the pre-tokenised codes rather than by tokenising twice: the
    conversion is one dictionary lookup per event, so the untruncated length is
    free where a second full ``__call__`` would not be.

    Args:
        tokenizer: A loaded ``CLMBRTokenizer``.
        events: List of ``hf_ehr.config.Event``.

    Returns:
        Integer token count.
    """
    return len(tokenizer.convert_events_to_tokens(events))


def embed_batch(tokenizer, model, batch_of_events, context, poolings):
    """One forward pass, every requested readout of it.

    The whole reason ``registry.extraction_key`` exists: ``last`` and ``mean``
    are two reductions of the tensor this already computed, so asking for both
    costs a second reduction rather than a second pass.

    Args:
        tokenizer: A loaded ``CLMBRTokenizer``.
        model: The loaded model, in eval mode.
        batch_of_events: List of per-patient ``Event`` lists.
        context: The cell's trained context window, used as ``max_length``.
        poolings: Readout names to compute.

    Returns:
        Tuple of ``({pooling: [B, d] array}, [B] untruncated token counts)``.
    """
    import torch

    untruncated = [count_tokens(tokenizer, events) for events in batch_of_events]
    encoded = tokenizer(
        batch_of_events,
        truncation=True,
        max_length=context,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
    hidden = outputs.hidden_states[-1].to(torch.float32).numpy()
    mask = encoded["attention_mask"].numpy()
    return {name: pool(hidden, mask, name) for name in poolings}, untruncated


def extract(database, index, cell, poolings, batch_size=BATCH_SIZE, progress=None):
    """Embed every anchor in ``index`` under one cell.

    Args:
        database: An open ``meds_reader`` database.
        index: DataFrame with ``person_id`` and ``cutoff`` columns, one row per
            anchor. ``cutoff`` may be ``NaT`` to keep a patient's whole record.
        cell: The ``registry.Cell`` to run.
        poolings: Readout names to emit, one matrix each.
        batch_size: Rows per forward pass.
        progress: Optional callable invoked with the number of rows completed.

    Returns:
        Tuple of ``({pooling: DataFrame}, provenance DataFrame)``. Each matrix
        carries ``person_id`` and ``cutoff`` alongside its ``dim_*`` columns;
        the provenance frame carries the same keys plus ``truncation_report``'s
        columns.
    """
    tokenizer, model = load_extractor(cell)
    events_by_patient = _events_by_patient(database, index)

    vectors = {name: [] for name in poolings}
    untruncated = []
    for start in range(0, len(index), batch_size):
        rows = index.iloc[start : start + batch_size]
        batch = [
            to_model_events(
                events_until(events_by_patient[row.person_id], _cutoff(row.cutoff))
            )
            for row in rows.itertuples()
        ]
        pooled, lengths = embed_batch(tokenizer, model, batch, cell.context, poolings)
        for name in poolings:
            vectors[name].append(pooled[name])
        untruncated.extend(lengths)
        if progress is not None:
            progress(min(start + batch_size, len(index)))

    keys = index[["person_id", "cutoff"]].reset_index(drop=True)
    matrices = {
        name: pd.concat([keys, _as_frame(np.concatenate(parts))], axis=1)
        for name, parts in vectors.items()
    }
    provenance = pd.concat([keys, truncation_report(untruncated, cell.context)], axis=1)
    return matrices, provenance


def extraction_record(cell, index, provenance, seconds):
    """The ``extraction.json`` payload: what ran, over what, and at what cost.

    Provenance rather than results, and written beside the matrices because a
    matrix whose truncation behaviour has to be reconstructed later is a matrix
    nobody can stratify.

    Args:
        cell: The ``registry.Cell`` that ran.
        index: The anchor index it ran over.
        provenance: The per-anchor frame from ``extract``.
        seconds: Wall clock for the run.

    Returns:
        JSON-serialisable dict.
    """
    return {
        "cell_id": cell.id,
        "source": cell.source,
        "context": cell.context,
        "truncation_side": TRUNCATION_SIDE,
        "n_anchors": len(index),
        "n_patients": int(index["person_id"].nunique()),
        # An anchor with no history before it embeds as a single PAD token, which
        # is a real vector that means nothing. It is legitimate at a firewalled
        # anchor and never legitimate in bulk, so it is counted here rather than
        # left to be noticed downstream as a suspiciously tight cluster.
        "n_empty_histories": int((provenance["n_tokens_untruncated"] == 0).sum()),
        "truncated_share": float(provenance["truncated"].mean()),
        "median_tokens_untruncated": float(provenance["n_tokens_untruncated"].median()),
        "median_covered": float(provenance["covered"].median()),
        "seconds": round(float(seconds), 1),
    }


def timed(function, *args, **kwargs):
    """Run ``function`` and return ``(result, seconds)``."""
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def _events_by_patient(database, index):
    """Read each patient's events once, however many anchors they carry.

    The lab tasks label the same patient at hundreds of prediction times, so
    re-reading per anchor would dominate the ``perlabel`` run.
    """
    return {
        person_id: patient_events(database, person_id)
        for person_id in index["person_id"].unique()
    }


def _as_frame(matrix):
    """``[N, d]`` array to a ``dim_0 .. dim_{d-1}`` frame."""
    return pd.DataFrame(matrix, columns=[f"dim_{i}" for i in range(matrix.shape[1])])
