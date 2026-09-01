"""The anchor indices: dedup, one-row-per-patient, and the levels that must not mix.

Pure pandas throughout -- `perlabel_index` is driven through a stubbed
`load_labels` rather than the licensed label files, so this runs anywhere. The
counts the Slurm sizing depends on are asserted against the real files in the
`integration` test at the bottom.
"""

import pandas as pd
import pytest

from embedding_space_exploration.data_management import anchors as anchors_module
from embedding_space_exploration.data_management.anchors import (
    LAB_TASKS,
    SCOUT_TASKS,
    build_index,
    lastevent_index,
    perlabel_index,
    restrict_to_known,
)
from embedding_space_exploration.data_management.ehrshot import ASSETS, TASKS


def _labels(pairs, task="t"):
    return pd.DataFrame(
        {
            "person_id": [p for p, _ in pairs],
            "prediction_time": pd.to_datetime([t for _, t in pairs]),
            "y": 0,
            "label_type": "boolean",
            "task": task,
        }
    )


@pytest.fixture
def stub_labels(monkeypatch):
    def install(by_task):
        monkeypatch.setattr(
            anchors_module, "load_labels", lambda task: _labels(by_task[task], task)
        )

    return install


def test_the_same_patient_at_the_same_instant_is_one_anchor(stub_labels):
    # Two tasks labelling one patient at one time want one vector, not two. The
    # real files hold 1,152,379 label rows over 381,522 distinct anchors, so
    # this dedup is a 3x saving before any scoping choice is made.
    stub_labels({"a": [(1, "2020-01-01"), (2, "2020-01-01")], "b": [(1, "2020-01-01")]})

    index = perlabel_index(("a", "b"))

    assert len(index) == 2
    assert list(index.columns) == ["person_id", "cutoff"]


def test_one_patient_at_two_times_is_two_anchors(stub_labels):
    stub_labels({"a": [(1, "2020-01-01"), (1, "2021-06-30")]})

    assert len(perlabel_index(("a",))) == 2


def test_the_scout_and_the_lab_tasks_partition_the_benchmark():
    # Derived by prefix rather than listed, so a task added to `TASKS` cannot
    # fall out of both buckets and be silently dropped from every anchor level.
    assert set(SCOUT_TASKS) | set(LAB_TASKS) == set(TASKS)
    assert not set(SCOUT_TASKS) & set(LAB_TASKS)
    assert len(SCOUT_TASKS) == 9
    assert len(LAB_TASKS) == 5


def test_lastevent_carries_the_real_timestamp_not_nat():
    # A stored NaT keeps the whole record too, but makes every row's key
    # identical, so a later join against the `shared` index cannot be checked.
    timeline = pd.DataFrame(
        {
            "person_id": [7, 3],
            "last_event": pd.to_datetime(["2022-03-05", "2019-11-02"]),
            "n_events": [10, 20],
        }
    )

    index = lastevent_index(timeline)

    assert list(index.columns) == ["person_id", "cutoff"]
    assert index["person_id"].tolist() == [3, 7]
    assert index["cutoff"].notna().all()
    assert index.loc[index["person_id"] == 7, "cutoff"].iloc[0] == pd.Timestamp(
        "2022-03-05"
    )


def test_the_shared_anchor_refuses_rather_than_guessing():
    # It needs the outcome window from decision B1. Returning *something* here
    # is how a placeholder window ends up in a published number.
    with pytest.raises(ValueError, match="B1"):
        build_index("shared")


def test_an_unknown_level_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="unknown anchor level"):
        build_index("wherever")


def test_lastevent_needs_the_timeline():
    with pytest.raises(ValueError, match="timeline"):
        build_index("lastevent")


@pytest.mark.integration
@pytest.mark.skipif(not ASSETS.exists(), reason="EHRSHOT assets not found")
def test_the_scout_is_a_small_fraction_of_the_full_perlabel_anchor():
    # The number the whole Slurm sizing rests on. If a future EHRSHOT release
    # changes it, the walltimes in `hpc/spartan/extract.slurm` are wrong and
    # this is where that surfaces.
    scout = perlabel_index(SCOUT_TASKS)
    full = perlabel_index(TASKS)

    assert len(scout) == 14_204
    assert len(full) == 381_522
    assert len(scout) / len(full) < 0.05


def _timeline(person_ids):
    return pd.DataFrame(
        {
            "person_id": person_ids,
            "last_event": pd.to_datetime(["2021-06-01"] * len(person_ids)),
        }
    )


def test_labelled_patients_absent_from_the_extract_are_dropped():
    # The failure this exists to prevent: five patients the EHRSHOT label files
    # name and the staged MEDS extract does not carry. `meds_reader` raises
    # KeyError on the first one, and it killed four cells at 14,000 of 14,204
    # rows with nothing written. Ten anchors are not worth a matrix.
    index = pd.DataFrame(
        {
            "person_id": [3, 999, 3],
            "cutoff": pd.to_datetime(["2021-01-01", "2021-02-02", "2021-03-03"]),
        }
    )

    kept, dropped = restrict_to_known(index, _timeline([3]))

    assert dropped == 1
    assert kept["person_id"].tolist() == [3, 3]


def test_the_filter_reindexes_so_the_vectors_stay_aligned():
    # `extract_resumable` concatenates the keys with positionally-produced
    # vectors, so a survivor that kept its pre-filter label misaligns the matrix.
    index = pd.DataFrame(
        {"person_id": [999, 3], "cutoff": pd.to_datetime(["2021-01-01", "2021-02-02"])}
    )

    kept, _ = restrict_to_known(index, _timeline([3]))

    assert kept.index.tolist() == [0]


def test_an_extract_carrying_every_labelled_patient_drops_nothing():
    index = pd.DataFrame(
        {"person_id": [3, 7], "cutoff": pd.to_datetime(["2021-01-01", "2021-02-02"])}
    )

    kept, dropped = restrict_to_known(index, _timeline([3, 7, 11]))

    assert dropped == 0
    assert len(kept) == 2


def test_build_index_filters_the_label_levels(stub_labels):
    stub_labels(
        {
            task: ([(3, "2021-01-01"), (999, "2021-02-02")] if i == 0 else [])
            for i, task in enumerate(SCOUT_TASKS)
        }
    )

    index, dropped = build_index("perlabel-scout", timeline=_timeline([3]))

    assert dropped == 1
    assert index["person_id"].tolist() == [3]


def test_lastevent_is_filtered_by_construction():
    index, dropped = build_index("lastevent", timeline=_timeline([3, 7]))

    assert dropped == 0
    assert index["person_id"].tolist() == [3, 7]


def test_the_perlabel_levels_need_the_timeline_to_filter_against():
    with pytest.raises(ValueError, match="timeline"):
        build_index("perlabel-scout")
