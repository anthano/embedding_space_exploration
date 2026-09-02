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

# What `truncation_report` produces, named once so the journal can select them
# back out without reconstructing an empty frame to ask.
TRUNCATION_COLUMNS = (
    "n_tokens_untruncated",
    "n_tokens_seen",
    "truncated",
    "covered",
)


def resolve_device(preference=None):
    """The device to run on: an explicit choice, else the best one present.

    CUDA first, then Apple's MPS, then CPU. Resolved rather than hard-coded
    because the same cell has to run on a laptop for a smoke test and on a
    cluster for the real pass, and the measured gap between those is the
    difference between a weekend and a quarter.

    Args:
        preference: An explicit device string, or ``None`` to auto-detect.

    Returns:
        Device string.
    """
    if preference is not None:
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        DataFrame with the ``TRUNCATION_COLUMNS``.
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


def load_extractor(cell, device=None):
    """Load one cell's tokeniser and backbone, with truncation pinned to the tail.

    Imported lazily and kept to one function so the pure half above stays
    runnable with neither ``torch`` nor ``hf_ehr`` installed.

    Returns the **backbone**, not the causal-LM wrapper. The LM head projects
    every position onto a 39,818-token vocabulary to produce logits this never
    reads: at batch 8 and context 4096 that materialises a
    ``[8, 4096, 39818]`` float32 tensor -- 5.2 GB, on a machine with 8 GB of
    unified memory -- purely to be discarded. ``output_hidden_states=True``
    compounds it by retaining all 13 layers when only the last is wanted.

    It has to be reached as ``model.base_model`` rather than by loading
    ``AutoModel`` directly. The checkpoints store backbone weights under a
    ``transformer.`` prefix, so ``AutoModel.from_pretrained`` mismatches the
    prefix, **silently reinitialises the word-embedding table** and warns rather
    than raises -- measured max abs difference 9.13 against the correct vectors,
    which is a random embedding table rather than a numerical wobble. Every
    downstream number would have been computed on noise with nothing to show
    for it in a traceback.

    Args:
        cell: A ``registry.Cell`` with ``loader == "hf_ehr"``.
        device: Device string, or ``None`` to auto-detect.

    Returns:
        Tuple of ``(tokenizer, backbone, device)``, the backbone in eval mode.

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
    device = resolve_device(device)
    backbone = AutoModelForCausalLM.from_pretrained(cell.source).base_model
    backbone.eval()
    backbone.to(device)
    return tokenizer, backbone, device


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


def embed_batch(tokenizer, backbone, batch_of_events, context, poolings, device="cpu"):
    """One forward pass, every requested readout of it.

    The whole reason ``registry.extraction_key`` exists: ``last`` and ``mean``
    are two reductions of the tensor this already computed, so asking for both
    costs a second reduction rather than a second pass.

    Args:
        tokenizer: A loaded ``CLMBRTokenizer``.
        backbone: The loaded backbone, in eval mode and already on ``device``.
        batch_of_events: List of per-patient ``Event`` lists.
        context: The cell's trained context window, used as ``max_length``.
        poolings: Readout names to compute.
        device: Where the forward pass runs.

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
    mask_tensor = encoded["attention_mask"]
    with torch.no_grad():
        outputs = backbone(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=mask_tensor.to(device),
        )
    hidden = outputs.last_hidden_state.to(torch.float32).cpu().numpy()
    mask = mask_tensor.numpy()
    return {name: pool(hidden, mask, name) for name in poolings}, untruncated


def extract(
    database,
    index,
    cell,
    poolings,
    batch_size=BATCH_SIZE,
    progress=None,
    device=None,
):
    """Embed every anchor in ``index`` under one cell.

    Args:
        database: An open ``meds_reader`` database.
        index: DataFrame with ``person_id`` and ``cutoff`` columns, one row per
            anchor. ``cutoff`` may be ``NaT`` to keep a patient's whole record.
        cell: The ``registry.Cell`` to run.
        poolings: Readout names to emit, one matrix each.
        batch_size: Rows per forward pass.
        progress: Optional callable invoked with the number of rows completed.
        device: Device string, or ``None`` to auto-detect.

    Returns:
        Tuple of ``({pooling: DataFrame}, provenance DataFrame)``. Each matrix
        carries ``person_id`` and ``cutoff`` alongside its ``dim_*`` columns;
        the provenance frame carries the same keys plus ``truncation_report``'s
        columns.
    """
    index = ordered_index(index)
    tokenizer, backbone, device = load_extractor(cell, device)

    vectors = {name: [] for name in poolings}
    untruncated = []
    for rows, pooled, lengths in iter_batches(
        database, index, cell, poolings, tokenizer, backbone, device, batch_size
    ):
        for name in poolings:
            vectors[name].append(pooled[name])
        untruncated.extend(lengths)
        if progress is not None:
            progress(min(rows.index[-1] + 1, len(index)))

    keys = index[["person_id", "cutoff"]].reset_index(drop=True)
    matrices = {
        name: pd.concat([keys, _as_frame(np.concatenate(parts))], axis=1)
        for name, parts in vectors.items()
    }
    provenance = pd.concat([keys, truncation_report(untruncated, cell.context)], axis=1)
    return matrices, provenance


def ordered_index(index):
    """The one anchor order every run uses, fresh or resumed.

    Sorted by patient, then by cutoff. Two things depend on it.

    **Resume correctness.** A resumed run skips the first ``k`` rows and picks up
    at row ``k``; that only lands on the same anchors as a fresh run if the order
    is a function of the index rather than of how it happened to be built.

    **Reading each patient once.** Sorting by ``person_id`` makes one patient's
    anchors contiguous, which is what lets ``_PatientEventCache`` hold exactly
    one patient and still never re-read. It matters most where the cost is worst:
    at the ``perlabel`` anchor 6,275 patients carry 381,522 anchors, so the lab
    tasks label the same patient hundreds of times.

    Args:
        index: DataFrame with ``person_id`` and ``cutoff`` columns.

    Returns:
        The same rows, sorted, with a fresh ``RangeIndex``.
    """
    return index.sort_values(
        ["person_id", "cutoff"], kind="stable", na_position="first"
    ).reset_index(drop=True)


def iter_batches(
    database,
    index,
    cell,
    poolings,
    tokenizer,
    backbone,
    device,
    batch_size=BATCH_SIZE,
    start_row=0,
):
    """Yield ``(rows, {pooling: [B, d] array}, untruncated)`` per forward pass.

    The single place batches are formed, so the in-memory path and the
    incremental path cannot drift into batching differently -- which would make
    their numbers differ in the last bits for no reason a reader could see.

    Args:
        database: An open ``meds_reader`` database.
        index: An already-``ordered_index`` frame.
        cell: The ``registry.Cell`` being run.
        poolings: Readout names to compute.
        tokenizer: A loaded ``CLMBRTokenizer``.
        backbone: The loaded backbone, on ``device``.
        device: Where the forward pass runs.
        batch_size: Rows per forward pass.
        start_row: Row to resume from. Must be a multiple of ``batch_size`` for a
            resumed run to batch identically to a fresh one.

    Yields:
        Tuple of ``(rows, pooled, untruncated)`` for each batch.
    """
    events = _PatientEventCache(database)
    for start in range(start_row, len(index), batch_size):
        rows = index.iloc[start : start + batch_size]
        batch = [
            to_model_events(
                events_until(events.get(row.person_id), _cutoff(row.cutoff))
            )
            for row in rows.itertuples()
        ]
        pooled, lengths = embed_batch(
            tokenizer, backbone, batch, cell.context, poolings, device
        )
        yield rows, pooled, lengths


def extraction_record(cell, index, provenance, seconds, device=None):
    """The ``extraction.json`` payload: what ran, over what, and at what cost.

    Provenance rather than results, and written beside the matrices because a
    matrix whose truncation behaviour has to be reconstructed later is a matrix
    nobody can stratify.

    Args:
        cell: The ``registry.Cell`` that ran.
        index: The anchor index it ran over.
        provenance: The per-anchor frame from ``extract``.
        seconds: Wall clock for the run.
        device: The device the forward passes ran on.

    Returns:
        JSON-serialisable dict.
    """
    return {
        "cell_id": cell.id,
        "source": cell.source,
        "context": cell.context,
        "truncation_side": TRUNCATION_SIDE,
        # Recorded because it is not purely a performance choice: CPU, MPS and
        # CUDA do not agree bit-for-bit, so a cell re-run on different hardware
        # is a different matrix in the last decimals. Cheap to note now, and the
        # first thing to check if a determinism arm ever disagrees with itself.
        "device": device,
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


# ======================================================================================
# Incremental writing and resume
# ======================================================================================
# A cell at the perlabel anchor is a job measured in tens of hours. Writing only
# at the end means a job killed at hour 55 of 60 leaves nothing, so results are
# flushed to a journal as they are produced and a restart continues from where
# the journal ends.
#
# This is deliberately *not* the chunked design. There is one process, one file
# per cell at the end, and no merge step anybody has to get right -- the journal
# is an implementation detail that is consumed and deleted on completion.
#
# Which is exactly why a *finished* cell needs its own guard rather than the
# journal. The journal is gone once the matrix is written, so a re-submitted
# element finds nothing to resume from and, without `is_complete`, recomputes
# the whole cell and overwrites the matrix with a bitwise-identical copy. That
# is what `extract.slurm`'s re-submission strategy rests on: an array element is
# re-run to finish the cells that died, and the cells that did not must cost
# seconds. At the `perlabel` anchor, where an element is measured in tens of
# hours, redoing a finished one is the difference between a re-submission and a
# second full run.
#
# A resumed run is bitwise identical to a fresh one, which is the property that
# makes this safe to use by default. Blocks are whole numbers of batches, so a
# resume point is always a batch boundary, so every batch after it contains the
# same anchors it would have in a clean run. Nothing about the arithmetic
# depends on where the job died.

JOURNAL_DIR = "_journal"

# Batches per flush. At the default batch size this is 200 anchors, ~1.2 MB of
# float32 across two poolings -- small enough that losing one costs seconds of
# recomputation, large enough that the run is not dominated by file creation.
FLUSH_EVERY = 25


def journal_dir(out_dir):
    """Where a cell's in-progress blocks live."""
    from pathlib import Path

    return Path(out_dir) / JOURNAL_DIR


def read_journal(out_dir):
    """Every complete block written so far, in order.

    Blocks are written to a temporary name and renamed into place, so a torn
    block should not exist; it is still handled, because "should not" is not a
    guarantee when a scheduler sends SIGKILL. The first unreadable block and
    everything after it is discarded -- those anchors are simply recomputed.

    Args:
        out_dir: The cell's output directory.

    Returns:
        A ``pyarrow.Table``, or ``None`` when nothing has been written.
    """
    import pyarrow as pa

    directory = journal_dir(out_dir)
    if not directory.exists():
        return None
    blocks = []
    paths = sorted(directory.glob("*.arrow"))
    for position, path in enumerate(paths):
        try:
            with pa.ipc.open_file(path) as reader:
                blocks.append(reader.read_all())
        except (pa.ArrowInvalid, OSError):
            for stale in paths[position:]:
                stale.unlink(missing_ok=True)
            break
    return pa.concat_tables(blocks) if blocks else None


def is_complete(index, targets, work_dir):
    """Whether every target already holds a finished matrix for ``index``.

    The completion marker the journal cannot be. ``read_journal`` returns
    ``None`` both for a cell that has never run and for one that finished --
    the journal is deleted on success -- so resume alone cannot tell them
    apart, and a re-submitted array element would recompute a cell it already
    has.

    Deliberately conservative on all three counts:

    * **Every** target, not just ``work_dir``. The matrices are written in a
      loop, so a crash between the two ``to_parquet`` calls leaves one pooling
      complete and the other absent.
    * ``extraction.json`` too, which is written after every matrix. Without it
      there is nothing to return, and its absence means the run died in the
      last few lines.
    * No journal. A complete matrix beside a live journal means a later run
      started and died; recomputing is both correct and self-healing, where
      skipping would leave the journal behind for the next reader to puzzle
      over.

    Row counts come from the parquet footer rather than the data, so this is
    metadata reads and not 43 MB per cell.

    Args:
        index: The anchor index the cell would run over.
        targets: ``{pooling: output directory}``, as ``extract_resumable`` takes.
        work_dir: The directory the journal lives in.

    Returns:
        True when the cell can be skipped.
    """
    import pyarrow.parquet as pq

    if journal_dir(work_dir).exists():
        return False
    for directory in targets.values():
        matrix = directory / "embeddings.parquet"
        if not matrix.exists() or not (directory / "extraction.json").exists():
            return False
        if pq.ParquetFile(matrix).metadata.num_rows != len(index):
            return False
    return True


def write_block(out_dir, table):
    """Append one block to the journal, atomically.

    Written under a temporary name and renamed, because a reader that finds a
    half-written block cannot tell it from a complete one.

    Args:
        out_dir: The cell's output directory.
        table: The ``pyarrow.Table`` to append.
    """
    import pyarrow as pa

    directory = journal_dir(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    index = len(list(directory.glob("*.arrow")))
    final = directory / f"{index:06d}.arrow"
    staged = final.with_suffix(".arrow.tmp")
    with pa.ipc.new_file(staged, table.schema) as writer:
        writer.write_table(table)
    staged.rename(final)


def _block_table(rows, pooled, untruncated, context):
    """One block: the keys, every pooling's vectors, and the truncation columns.

    Held in one table rather than one per pooling so that a block is either
    present for all of them or absent for all of them. Two journals could be
    resumed at two different points, and the matrices would then disagree about
    which anchor each row is.
    """
    import pyarrow as pa

    frame = rows[["person_id", "cutoff"]].reset_index(drop=True)
    frame = pd.concat(
        [frame, truncation_report(untruncated, context)],
        axis=1,
    )
    for name, matrix in pooled.items():
        block = _as_frame(matrix)
        block.columns = [f"{name}__{column}" for column in block.columns]
        frame = pd.concat([frame, block], axis=1)
    return pa.Table.from_pandas(frame, preserve_index=False)


def _split_journal(table, poolings):
    """Journal table back into ``({pooling: matrix frame}, provenance frame)``."""
    frame = table.to_pandas()
    keys = frame[["person_id", "cutoff"]]
    matrices = {}
    for name in poolings:
        prefix = f"{name}__"
        columns = [column for column in frame.columns if column.startswith(prefix)]
        block = frame[columns].rename(
            columns={column: column[len(prefix) :] for column in columns}
        )
        matrices[name] = pd.concat([keys, block], axis=1)
    provenance = pd.concat([keys, frame[list(TRUNCATION_COLUMNS)]], axis=1)
    return matrices, provenance


def extract_resumable(
    database,
    index,
    cell,
    targets,
    batch_size=BATCH_SIZE,
    flush_every=FLUSH_EVERY,
    device=None,
    progress=None,
    provenance_extra=None,
):
    """Embed every anchor, flushing as it goes and resuming where it left off.

    The entry point a cluster job calls. One process, one pass, one
    ``embeddings.parquet`` per pooling at the end.

    Returns immediately when the cell is already finished (see ``is_complete``),
    without loading a model, and the returned record carries ``skipped``. That
    is what makes re-submitting a whole array cheap rather than a second full
    run.

    Args:
        database: An open ``meds_reader`` database.
        index: DataFrame with ``person_id`` and ``cutoff`` columns.
        cell: Any ``registry.Cell`` of the extraction group -- the poolings come
            from ``targets``, and every cell in a group shares the forward pass.
        targets: ``{pooling: output directory}``. One directory per cell id, so
            each is self-contained for everything downstream.
        batch_size: Rows per forward pass.
        flush_every: Batches per journal block.
        device: Device string, or ``None`` to auto-detect.
        progress: Optional callable invoked with the number of rows completed.
        provenance_extra: Optional fields merged into ``extraction.json``. For
            facts the caller knows and this function cannot see -- the anchor
            filter's drop count is one, and it belongs beside the matrix rather
            than only in a Slurm log that ages out.

    Returns:
        The ``extraction_record`` dict for the run.
    """
    import json

    started = time.perf_counter()
    index = ordered_index(index)
    poolings = tuple(targets)
    work_dir = next(iter(targets.values()))
    for directory in targets.values():
        directory.mkdir(parents=True, exist_ok=True)

    if is_complete(index, targets, work_dir):
        # Return the record the finished run wrote rather than a synthesised
        # one: its `seconds` and `device` describe the pass that actually
        # produced these matrices, and re-stamping them with this process's
        # wall clock would turn the only throughput measurement the project has
        # into a number about doing nothing.
        record = json.loads((work_dir / "extraction.json").read_text())
        record["skipped"] = True
        return record

    existing = read_journal(work_dir)
    done = 0 if existing is None else existing.num_rows
    if done % batch_size and done < len(index):
        # Only whole blocks are ever written, so this means the journal was
        # produced with a different batch size. Continuing would batch the
        # remaining anchors differently from a clean run, which is exactly the
        # reproducibility claim this design makes; start the cell over instead.
        raise ValueError(
            f"journal in {work_dir} holds {done} rows, not a multiple of "
            f"batch_size={batch_size}; delete {journal_dir(work_dir)} and re-run"
        )
    if done >= len(index):
        done = len(index)

    tokenizer, backbone, device = load_extractor(cell, device)
    pending = []
    for rows, pooled, lengths in iter_batches(
        database,
        index,
        cell,
        poolings,
        tokenizer,
        backbone,
        device,
        batch_size,
        start_row=done,
    ):
        pending.append(_block_table(rows, pooled, lengths, cell.context))
        if len(pending) >= flush_every:
            import pyarrow as pa

            write_block(work_dir, pa.concat_tables(pending))
            pending = []
        if progress is not None:
            progress(min(rows.index[-1] + 1, len(index)))
    if pending:
        import pyarrow as pa

        write_block(work_dir, pa.concat_tables(pending))

    table = read_journal(work_dir)
    if table is None or table.num_rows != len(index):
        got = 0 if table is None else table.num_rows
        raise RuntimeError(
            f"journal holds {got} rows for an index of {len(index)}; refusing to "
            "write a partial matrix"
        )
    matrices, provenance = _split_journal(table, poolings)
    for name, directory in targets.items():
        matrices[name].to_parquet(directory / "embeddings.parquet", index=False)
        provenance.to_parquet(directory / "truncation.parquet", index=False)

    record = extraction_record(
        cell, index, provenance, time.perf_counter() - started, device
    )
    record["resumed_from_row"] = int(done)
    record["batch_size"] = int(batch_size)
    record.update(provenance_extra or {})
    for directory in targets.values():
        # The layout is `cells/{cell_id}/`, so the directory names the cell this
        # matrix belongs to -- which is not the representative cell the group was
        # run under, and writing that one into every pooling's provenance would
        # label the mean-pool matrix with the last-token cell's id.
        payload = {**record, "cell_id": directory.name}
        (directory / "extraction.json").write_text(json.dumps(payload, indent=2))

    for path in journal_dir(work_dir).glob("*.arrow"):
        path.unlink()
    journal_dir(work_dir).rmdir()
    return record


class _PatientEventCache:
    """One patient's events, held until the next patient is asked for.

    Replaces a dict of every patient in the index. That dict read each patient
    once, which was the point, but it also held all of them at once: 6,275
    patients at a median 3,129 events is ~20M tuples, several GB of Python
    objects, reached before the first forward pass. Fine on a 495 GB node and
    not fine on a laptop, and the resumable path has to run in both.

    A single-entry cache is enough precisely because ``ordered_index`` sorts by
    ``person_id``, so a patient's anchors arrive together and each patient is
    still read exactly once. Bounded memory and the same number of reads.
    """

    def __init__(self, database):
        self._database = database
        self._person_id = None
        self._rows = None

    def get(self, person_id):
        """The patient's events, reading them only on a change of patient."""
        if person_id != self._person_id:
            self._person_id = person_id
            self._rows = patient_events(self._database, person_id)
        return self._rows


def _as_frame(matrix):
    """``[N, d]`` array to a ``dim_0 .. dim_{d-1}`` frame."""
    return pd.DataFrame(matrix, columns=[f"dim_{i}" for i in range(matrix.shape[1])])
