"""The Tier 1 design's invariants.

Not tests of behaviour -- ``registry`` is data and has almost none. These are the
checks that stop the design disagreeing with the documents it is derived from,
which is the failure mode that costs a re-extraction rather than a traceback.
"""

import pytest

from embedding_space_exploration.registry import (
    CELLS,
    CONTEXT_CLUES,
    ENCODER,
    ID_COLUMNS,
    MEAN_POOL_CELLS,
    MODEL_CELLS,
    SCALINGS,
    battery_dir,
    cell_dir,
    extraction_key,
    id_columns,
    poolings_by_extraction,
)


def test_model_cell_count_matches_the_design_freeze():
    """Section 3 calls it ~18 frozen cells; the count must be derived, not asserted."""
    assert len(MODEL_CELLS) == 18


def test_cell_ids_are_unique():
    """``CELLS`` is a dict comprehension, so a collision would silently drop a cell."""
    all_cells = (*MODEL_CELLS, *MEAN_POOL_CELLS)
    assert len({cell.id for cell in all_cells}) == len(all_cells)
    assert len(CELLS) == 38


def test_context_spine_is_four_families_by_four_lengths():
    """The paper's spine. A missing length silently weakens a replication to three."""
    assert len(CONTEXT_CLUES) == 16
    by_family = {}
    for cell in CONTEXT_CLUES:
        by_family.setdefault(cell.family, set()).add(cell.context)
    assert by_family == {
        "gpt": {512, 1024, 2048, 4096},
        "llama": {512, 1024, 2048, 4096},
        "mamba": {1024, 4096, 8192, 16384},
        "hyena": {1024, 4096, 8192, 16384},
    }


def test_encoder_has_no_last_token_cell():
    """Every position in a bidirectional encoder has seen everything, so a
    last-token read of one is not a pooling arm -- it is an arbitrary token.
    """
    assert ENCODER.pooling == "mean"
    assert f"{ENCODER.family}-{ENCODER.size}-{ENCODER.context}-last" not in CELLS


def test_every_pooling_arm_has_a_last_token_counterpart():
    """P4 is a *paired* contrast; an unpaired mean-pool cell measures nothing."""
    for cell in MEAN_POOL_CELLS:
        counterpart = f"{cell.family}-{cell.size}-{cell.context}-last"
        assert counterpart in CELLS, counterpart


def test_only_the_encoder_arm_eats_text():
    """``token_unit`` is what stops context being plotted on one axis across arms."""
    text_cells = [cell.id for cell in MODEL_CELLS if cell.token_unit == "text"]
    assert text_cells == [ENCODER.id]


def test_id_columns_cover_every_declared_column():
    columns = id_columns("ehrshot", "shared", "gpt-base-512-last", "spherical")
    assert tuple(columns) == ID_COLUMNS


@pytest.mark.parametrize("scaling", SCALINGS)
def test_anchors_never_share_a_directory(scaling):
    """The 2026-08-27 ledger entry turns on Y1 and Y2 never mixing anchors. A
    directory level makes that a missing file rather than a wrong number.
    """
    cell = "gpt-base-512-last"
    assert cell_dir("ehrshot", "shared", cell) != cell_dir("ehrshot", "perlabel", cell)
    assert battery_dir("ehrshot", "shared", cell, scaling) != battery_dir(
        "ehrshot", "perlabel", cell, scaling
    )


def test_pooling_variants_share_one_forward_pass():
    """The correction that dissolved the P4 subset: cells differing only in
    pooling are two reductions of one ``[1, T, d]`` tensor, not two passes.
    """
    groups = poolings_by_extraction()
    assert groups[("gpt", "base", 512)] == ("gpt-base-512-last", "gpt-base-512-mean")
    # 17 decoders + the encoder; baselines run no model and are excluded.
    assert len(groups) == 18
    assert sum(len(members) for members in groups.values()) == 35


def test_baselines_have_no_extraction_key():
    """They run no model, so they are not in the forward-pass DAG at all."""
    assert all(extraction_key(CELLS[name]) is None for name in ("tfidf", "clinical"))
