"""The reductions, the truncation accounting, and the one default that must not drift.

Everything here runs without ``torch``, ``hf_ehr`` or the licensed extract, which
is the point of the split in ``extraction``: the impure half is thin enough to
read, and the half where a mistake survives a green run is the half tested here.
"""

import json

import numpy as np
import pandas as pd
import pytest

from embedding_space_exploration.data_management import extraction as extraction_module
from embedding_space_exploration.data_management.extraction import (
    TRUNCATION_SIDE,
    _cutoff,
    extract_resumable,
    extraction_record,
    journal_dir,
    ordered_index,
    pool,
    pool_last,
    pool_mean,
    read_journal,
    resolve_device,
    truncation_report,
    write_block,
)
from embedding_space_exploration.data_management.timeline import (
    MEDS_READER_DIR,
    open_database,
    subject_ids,
)
from embedding_space_exploration.registry import CELLS


@pytest.fixture
def database():
    """An open extract that is closed again.

    ``SubjectDatabase`` holds worker resources and is a context manager for that
    reason. Leaking one per test wedged the *simulation* suite much later in the
    run -- a live reader plus a fork-based pool is a deadlock, and it presented
    as an unrelated test hanging rather than as anything pointing here.
    """
    with open_database() as opened:
        yield opened


@pytest.fixture
def padded():
    """Two sequences of length 3 and 1, right-padded to T=3."""
    hidden = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = np.array([[1, 1, 1], [1, 0, 0]])
    return hidden, mask


def test_last_token_reads_the_last_real_position_not_the_last_column(padded):
    """With right padding an unmasked ``[:, -1, :]`` reads PAD for every short row."""
    hidden, mask = padded
    assert pool_last(hidden, mask).tolist() == [[5.0, 6.0], [7.0, 8.0]]
    # The bug this guards against, stated so the difference is visible.
    assert hidden[:, -1, :].tolist() == [[5.0, 6.0], [0.0, 0.0]]


def test_mean_pool_divides_by_real_length_not_padded_length(padded):
    """An unmasked mean divides a real sum by a padded count and shrinks short rows."""
    hidden, mask = padded
    assert pool_mean(hidden, mask).tolist() == [[3.0, 4.0], [7.0, 8.0]]
    assert hidden.mean(axis=1).tolist() != pool_mean(hidden, mask).tolist()


def test_pooling_is_a_reduction_of_one_tensor(padded):
    """P4's cost claim: both readouts come off the same ``[B, T, d]`` array."""
    hidden, mask = padded
    for name in ("last", "mean"):
        assert pool(hidden, mask, name).shape == (2, 2)


def test_unknown_pooling_is_an_error(padded):
    hidden, mask = padded
    with pytest.raises(KeyError):
        pool(hidden, mask, "first")


def test_an_empty_history_does_not_divide_by_zero():
    """MEDS permits a patient with nothing before the anchor; the run must not die."""
    hidden = np.zeros((1, 2, 3))
    mask = np.zeros((1, 2), dtype=int)
    assert not np.isnan(pool_mean(hidden, mask)).any()


def test_truncation_report_flags_only_what_exceeds_the_window():
    report = truncation_report([100, 512, 513, 4000], context=512)
    assert report["truncated"].tolist() == [False, False, True, True]
    assert report["n_tokens_seen"].tolist() == [100, 512, 512, 512]


def test_covered_is_the_share_of_record_the_model_saw():
    """The quantity comparable across ``token_unit``; the nominal window is not."""
    report = truncation_report([256, 1024], context=512)
    assert report["covered"].tolist() == [1.0, 0.5]


def test_covered_is_defined_for_a_zero_length_history():
    assert truncation_report([0], context=512)["covered"].tolist() == [1.0]


def test_truncation_keeps_the_tail():
    """The anchor is the *end* of the history, so the cut must drop the oldest tokens.

    HuggingFace defaults to ``truncation_side='right'``, which keeps the record's
    opening and reads a "last token" nowhere near the anchor -- measured at cosine
    0.52 against the correct vector on ``gpt-base-512-clmbr``. It would also make
    every cell in a family see the same opening tokens, which builds P1's null
    into the design. Asserted rather than commented because a ``transformers``
    bump could flip the default back.
    """
    assert TRUNCATION_SIDE == "left"


def test_extraction_record_carries_the_stratifier_and_the_cut():
    cell = CELLS["gpt-base-512-last"]
    index = pd.DataFrame({"person_id": [1, 1, 2], "cutoff": pd.NaT})
    provenance = truncation_report([100, 600, 1024], context=cell.context)
    record = extraction_record(cell, index, provenance, seconds=12.34)
    assert record["truncation_side"] == "left"
    assert record["n_anchors"] == 3
    assert record["n_patients"] == 2
    assert record["truncated_share"] == pytest.approx(2 / 3)
    assert record["median_covered"] == pytest.approx(512 / 600)


@pytest.mark.parametrize("missing", [None, pd.NaT, np.nan])
def test_a_missing_cutoff_normalises_to_none(missing):
    """``events_until`` reads ``None`` as "keep everything"; a frame supplies ``NaT``.

    Every ``event_time <= NaT`` is ``False``, so an unnormalised missing cutoff
    empties the history instead of keeping it and the patient is embedded as one
    PAD token -- five identical vectors and a zero token count, with no error.
    """
    assert _cutoff(missing) is None


def test_a_real_cutoff_passes_through():
    stamp = pd.Timestamp("2015-06-01")
    assert _cutoff(stamp) == stamp


def test_extraction_record_counts_empty_histories():
    """Legitimate at a firewalled anchor, never legitimate in bulk."""
    cell = CELLS["gpt-base-512-last"]
    index = pd.DataFrame({"person_id": [1, 2, 3], "cutoff": pd.NaT})
    provenance = truncation_report([0, 0, 900], context=cell.context)
    assert extraction_record(cell, index, provenance, 1.0)["n_empty_histories"] == 2


@pytest.mark.parametrize("preference", ["cpu", "mps", "cuda", "cuda:1"])
def test_an_explicit_device_is_honoured_without_importing_torch(preference):
    """A cluster run names its device; only auto-detection needs to ask torch."""
    assert resolve_device(preference) == preference


def test_extraction_record_notes_the_device():
    """CPU, MPS and CUDA do not agree bit-for-bit, so the matrix depends on it."""
    cell = CELLS["gpt-base-512-last"]
    index = pd.DataFrame({"person_id": [1], "cutoff": pd.NaT})
    provenance = truncation_report([900], context=cell.context)
    record = extraction_record(cell, index, provenance, 1.0, device="mps")
    assert record["device"] == "mps"


# ======================================================================================
# Ordering, caching and the journal
# ======================================================================================


def test_ordered_index_groups_a_patient_together():
    """A patient's anchors must be contiguous or the one-entry cache re-reads."""
    index = pd.DataFrame(
        {
            "person_id": [3, 1, 3, 1, 2],
            "cutoff": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2019-01-01", "2018-01-01", "2021-01-01"]
            ),
        }
    )
    assert ordered_index(index)["person_id"].tolist() == [1, 1, 2, 3, 3]


def test_ordered_index_is_a_function_of_the_rows_not_their_arrival_order():
    """Resume skips the first k rows, which is only correct if the order is stable."""
    index = pd.DataFrame(
        {"person_id": [3, 1, 2], "cutoff": pd.to_datetime(["2020-01-01"] * 3)}
    )
    shuffled = index.iloc[[2, 0, 1]]
    assert ordered_index(index).equals(ordered_index(shuffled))


def test_patient_events_are_read_once_per_patient(monkeypatch):
    """The whole reason for sorting by person_id."""
    reads = []

    monkeypatch.setattr(
        extraction_module, "patient_events", lambda db, pid: reads.append(pid) or [pid]
    )
    cache = extraction_module._PatientEventCache(object())
    for person_id in (1, 1, 1, 2, 2, 3):
        cache.get(person_id)
    assert reads == [1, 2, 3]


def test_a_journal_round_trips(tmp_path):
    import pyarrow as pa

    table = pa.table({"person_id": [1, 2], "value": [0.5, 1.5]})
    write_block(tmp_path, table)
    assert read_journal(tmp_path).num_rows == 2


def test_journal_blocks_accumulate_in_order(tmp_path):
    import pyarrow as pa

    for value in (10, 20, 30):
        write_block(tmp_path, pa.table({"person_id": [value]}))
    assert read_journal(tmp_path)["person_id"].to_pylist() == [10, 20, 30]


def test_an_empty_journal_reads_as_nothing(tmp_path):
    assert read_journal(tmp_path) is None


def test_a_torn_block_and_everything_after_it_is_discarded(tmp_path):
    """A SIGKILL mid-write must cost recomputation, never a silently short matrix."""
    import pyarrow as pa

    for value in (1, 2, 3):
        write_block(tmp_path, pa.table({"person_id": [value]}))
    blocks = sorted(journal_dir(tmp_path).glob("*.arrow"))
    blocks[1].write_bytes(b"not an arrow file")

    assert read_journal(tmp_path)["person_id"].to_pylist() == [1]
    # The torn block and its successor are gone, so their rows get recomputed.
    assert len(list(journal_dir(tmp_path).glob("*.arrow"))) == 1


@pytest.mark.integration
@pytest.mark.skipif(
    not MEDS_READER_DIR.exists(),
    reason="meds_reader extract not found (set EHRSHOT_ROOT)",
)
def test_a_resumed_run_is_bitwise_identical_to_a_fresh_one(tmp_path, database):
    """The property that makes incremental writing safe to use by default.

    Blocks hold whole batches, so a resume point is always a batch boundary and
    every batch after it holds the anchors it would have held anyway. If this
    ever fails, the run stopped being reproducible and the fix is to start the
    cell over, not to accept the matrix.
    """
    cell = CELLS["gpt-base-512-last"]
    index = pd.DataFrame(
        {"person_id": sorted(subject_ids(database))[:8], "cutoff": pd.NaT}
    )

    clean = tmp_path / "clean"
    extract_resumable(
        database, index, cell, {"last": clean}, batch_size=2, flush_every=1
    )

    # Now the same run into a fresh directory, killed halfway: blocks are on
    # disk, the matrix is not, and the journal survives.
    resumed = tmp_path / "resumed"

    class Killed(Exception):
        pass

    def die_after_four(completed):
        if completed >= 4:
            raise Killed

    with pytest.raises(Killed):
        extract_resumable(
            database,
            index,
            cell,
            {"last": resumed},
            batch_size=2,
            flush_every=1,
            progress=die_after_four,
        )
    assert not (resumed / "embeddings.parquet").exists()
    assert read_journal(resumed).num_rows == 4

    record = extract_resumable(
        database, index, cell, {"last": resumed}, batch_size=2, flush_every=1
    )

    assert record["resumed_from_row"] == 4
    left = pd.read_parquet(clean / "embeddings.parquet")
    right = pd.read_parquet(resumed / "embeddings.parquet")
    pd.testing.assert_frame_equal(left, right)


@pytest.mark.integration
@pytest.mark.skipif(
    not MEDS_READER_DIR.exists(),
    reason="meds_reader extract not found (set EHRSHOT_ROOT)",
)
def test_the_journal_is_cleared_once_the_matrix_is_written(tmp_path, database):
    """The journal is an implementation detail, not an artifact to reason about."""
    cell = CELLS["gpt-base-512-last"]
    index = pd.DataFrame(
        {"person_id": sorted(subject_ids(database))[:4], "cutoff": pd.NaT}
    )
    out = tmp_path / "cells" / "gpt-base-512-last"
    extract_resumable(database, index, cell, {"last": out}, batch_size=2)

    assert not journal_dir(out).exists()
    assert (out / "embeddings.parquet").exists()
    assert json.loads((out / "extraction.json").read_text())["cell_id"] == (
        "gpt-base-512-last"
    )
