"""The reductions, the truncation accounting, and the one default that must not drift.

Everything here runs without ``torch``, ``hf_ehr`` or the licensed extract, which
is the point of the split in ``extraction``: the impure half is thin enough to
read, and the half where a mistake survives a green run is the half tested here.
"""

import numpy as np
import pandas as pd
import pytest

from embedding_space_exploration.data_management.extraction import (
    TRUNCATION_SIDE,
    _cutoff,
    extraction_record,
    pool,
    pool_last,
    pool_mean,
    truncation_report,
)
from embedding_space_exploration.registry import CELLS


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
