import numpy as np
import pandas as pd
import pytest

from embedding_space_exploration.data_management.timeline import (
    events_until,
    summarise_patient,
)


def _rows():
    """A patient: demographics at birth, then three clinical events."""
    birth = pd.Timestamp("1950-03-04")
    return [
        (birth, "MEDS_BIRTH", None),
        (birth, "Gender/F", None),
        (birth, "Race/5", None),
        (birth, "Ethnicity/Hispanic", None),
        (pd.Timestamp("2010-01-01"), "SNOMED/1234", None),
        (pd.Timestamp("2015-06-15"), "LOINC/29463-7", 70.5),
        (pd.Timestamp("2020-12-31"), "RxNorm/999", None),
    ]


def test_the_cut_is_inclusive_of_the_anchor():
    # A prediction time is the moment after which nothing is known, so an event
    # stamped exactly at the anchor precedes the outcome and is kept. Excluding
    # it would silently discard the most recent event at every anchor.
    kept = events_until(_rows(), pd.Timestamp("2015-06-15"))
    assert [code for _, code, _ in kept][-1] == "LOINC/29463-7"


def test_nothing_after_the_cut_survives():
    kept = events_until(_rows(), pd.Timestamp("2016-01-01"))
    assert all(time <= pd.Timestamp("2016-01-01") for time, _, _ in kept)
    assert len(kept) == 6


def test_a_cut_before_any_clinical_event_keeps_only_demographics():
    # The firewall's limiting case: anchoring before the record starts must not
    # leak a later event through.
    kept = events_until(_rows(), pd.Timestamp("1999-01-01"))
    assert {code for _, code, _ in kept} == {
        "MEDS_BIRTH",
        "Gender/F",
        "Race/5",
        "Ethnicity/Hispanic",
    }


def test_no_cutoff_keeps_the_whole_timeline():
    assert len(events_until(_rows(), None)) == len(_rows())


def test_undated_events_survive_every_cut():
    # MEDS permits timeless events and AoU writes them. They carry demographics,
    # so dropping them at an early anchor would silently vary the covariates by
    # anchor rather than by patient.
    rows = [(None, "Gender/F", None), (pd.Timestamp("2020-01-01"), "SNOMED/1", None)]
    assert len(events_until(rows, pd.Timestamp("2000-01-01"))) == 1


def test_the_record_starts_at_a_real_encounter_not_at_birth():
    # Demographics carry the birth timestamp. Counting them as record events
    # would start every history at birth and turn observation span into age.
    summary = summarise_patient(7, _rows())
    assert summary["first_event"] == pd.Timestamp("2010-01-01")
    assert summary["last_event"] == pd.Timestamp("2020-12-31")
    assert summary["n_events"] == 3


def test_observation_span_and_age_are_different_quantities():
    summary = summarise_patient(7, _rows())
    assert summary["observation_years"] == pytest.approx(10.99, abs=0.01)
    assert summary["age_at_last_event"] == pytest.approx(70.83, abs=0.01)


def test_demographics_are_read_off_the_birth_block():
    summary = summarise_patient(7, _rows())
    assert (summary["gender"], summary["race"], summary["ethnicity"]) == (
        "F",
        "5",
        "Hispanic",
    )


def test_a_missing_demographic_is_none_rather_than_an_error():
    # Race is absent for roughly a fifth of the cohort.
    rows = [r for r in _rows() if not r[1].startswith("Race/")]
    assert summarise_patient(7, rows)["race"] is None


def test_a_patient_with_no_clinical_events_summarises_rather_than_raising():
    rows = [r for r in _rows() if r[1].startswith(("MEDS_BIRTH", "Gender/"))]
    summary = summarise_patient(7, rows)
    assert summary["n_events"] == 0
    assert summary["first_event"] is None
    assert np.isnan(summary["observation_years"])
