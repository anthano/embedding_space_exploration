"""Patient timelines from the EHRSHOT ``meds_reader`` extract.

The path from stored events to something a model can consume, and the source of
the per-patient facts several later checks need: history length (A3, and the
length-stratified reporting in section 7), event count (the nuisance-only
baseline and C3's confound), demographics (C1's covariate assembler).

Split in two on purpose. Reading events needs the extract; turning them into
model input needs ``hf_ehr``; deciding *which* events to keep needs neither. So
``events_until`` -- the part that carries the temporal firewall, and the part
most worth testing -- is pure, operates on tuples, and can be exercised with no
licensed data and no optional dependency installed.

The generalisation ``allofus``'s oracle script did not need: that script cut a
patient's history at a label's ``prediction_time`` and fed it straight to a
tokeniser. The battery needs the same cut at an anchor that is not a label time,
and needs per-patient summaries the probe never asked for.
"""

import numpy as np
import pandas as pd

from embedding_space_exploration.config import EHRSHOT_ROOT

MEDS_READER_DIR = EHRSHOT_ROOT / "meds_reader_omop_ehrshot"

# The demographic block. Every subject carries these stamped at the birth
# timestamp rather than at a clinical encounter, so counting them as record
# events would make every patient's history begin at birth and turn observation
# span into age. Held apart from the clinical record for that reason, not
# discarded -- they are C1's covariates.
BIRTH_CODE = "MEDS_BIRTH"
DEMOGRAPHIC_PREFIXES = ("Gender", "Race", "Ethnicity")


def open_database(path=MEDS_READER_DIR):
    """Open the ``meds_reader`` extract.

    Imported lazily: the extract is licensed data that need not be present for
    the label layer, the probe or the oracle to run.

    Args:
        path: Directory of the ``meds_reader`` database.

    Returns:
        An open ``meds_reader.SubjectDatabase``.
    """
    import meds_reader

    return meds_reader.SubjectDatabase(str(path))


def subject_ids(database):
    """Person ids the extract actually holds.

    EHRSHOT's label files reference a handful of patients absent from the
    extract (6,731 subjects against 6,739 in the split map), so callers must
    filter rather than index blindly.

    Args:
        database: An open database.

    Returns:
        Set of integer person ids.
    """
    return {int(subject_id) for subject_id in database}


def patient_events(database, person_id):
    """One patient's events as ``(time, code, numeric_value)``, time-sorted.

    Pulled once per patient and reused across every cut of that patient's
    history -- the lab tasks label the same patient at hundreds of prediction
    times, so re-reading per cut would dominate the run.

    Events with no timestamp sort to the front. This extract has none (its
    demographics carry the birth timestamp instead), but MEDS permits them and
    AoU writes them, so the ordering is kept rather than assumed away.

    Args:
        database: An open database.
        person_id: The patient to read.

    Returns:
        List of ``(time, code, numeric_value)``, ascending in time.
    """
    subject = database[int(person_id)]
    rows = [
        (event.time, event.code, getattr(event, "numeric_value", None))
        for event in subject.events
    ]
    return sorted(rows, key=lambda row: (row[0] is not None, row[0]))


def events_until(rows, cutoff):
    """The history available at ``cutoff``, inclusive.

    **This is the temporal firewall.** Section 7 anchors every embedding at the
    last event before a declared outcome window so that the label defined inside
    that window is not already present in the history the model sees; this
    function is where that guarantee is enforced. Inclusive of ``cutoff``
    because a prediction time is the moment *after* which nothing is known, and
    events stamped at exactly that moment precede the outcome.

    Args:
        rows: Output of ``patient_events``.
        cutoff: Timestamp to cut at, inclusive. ``None`` keeps everything.

    Returns:
        The prefix of ``rows`` at or before ``cutoff``, in the same order.
    """
    if cutoff is None:
        return list(rows)
    return [row for row in rows if row[0] is None or row[0] <= cutoff]


def to_model_events(rows):
    """Convert timeline tuples into the ``Event`` list a tokeniser consumes.

    Assumes ``meds_reader`` code strings already match the CLMBR tokeniser's
    vocabulary form (``SNOMED/..``, ``LOINC/..``, ``RxNorm/..``), which holds for
    this OMOP extract. A dataset where it does not -- AoU's MEDS demographic
    codes are the known case -- remaps here rather than downstream.

    Args:
        rows: Output of ``patient_events`` or ``events_until``.

    Returns:
        List of ``hf_ehr.config.Event``.
    """
    from hf_ehr.config import Event

    return [
        Event(code=code, value=None if _missing(value) else float(value))
        for _, code, value in rows
    ]


def summarise_patient(person_id, rows):
    """Per-patient facts the battery and the baselines need.

    The clinical record is measured with the demographic block excluded, so
    ``first_event`` is a real encounter and ``observation_years`` is time under
    observation rather than age.

    Args:
        person_id: The patient.
        rows: Output of ``patient_events``.

    Returns:
        Dict with birth date, clinical record bounds, ``n_events``,
        ``observation_years`` and the three demographic fields.
    """
    birth = next((time for time, code, _ in rows if code == BIRTH_CODE), None)
    demographics = {
        prefix.lower(): next(
            (
                code.split("/", 1)[1]
                for _, code, _ in rows
                if code.startswith(f"{prefix}/")
            ),
            None,
        )
        for prefix in DEMOGRAPHIC_PREFIXES
    }
    clinical = [row for row in rows if not _is_demographic(row[1])]
    first = clinical[0][0] if clinical else None
    last = clinical[-1][0] if clinical else None
    return {
        "person_id": int(person_id),
        "birth_date": birth,
        "first_event": first,
        "last_event": last,
        "n_events": len(clinical),
        "observation_years": _years_between(first, last),
        "age_at_last_event": _years_between(birth, last),
        **demographics,
    }


def summarise_cohort(database, person_ids=None):
    """``summarise_patient`` over the whole extract.

    Args:
        database: An open database.
        person_ids: Patients to summarise. Defaults to every subject present.

    Returns:
        Frame with one row per patient.
    """
    ids = sorted(subject_ids(database)) if person_ids is None else list(person_ids)
    return pd.DataFrame(
        [
            summarise_patient(person_id, patient_events(database, person_id))
            for person_id in ids
        ]
    )


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _missing(value):
    """Whether a numeric value is absent, covering both ``None`` and NaN."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def _is_demographic(code):
    """Whether a code belongs to the birth-stamped demographic block."""
    return code == BIRTH_CODE or code.startswith(
        tuple(f"{prefix}/" for prefix in DEMOGRAPHIC_PREFIXES)
    )


def _years_between(start, end):
    """Years from ``start`` to ``end``, or NaN if either is missing."""
    if start is None or end is None:
        return np.nan
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
