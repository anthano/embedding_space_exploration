"""Which ``(person_id, cutoff)`` pairs get embedded, per anchor level.

The missing half of the extraction path. ``extraction.extract_resumable`` takes an
index of anchors and turns it into a matrix; nothing built that index, so nothing
could run it outside the tests. This is that builder, and it is pure pandas over
the label files and the timeline summary -- no model, no ``meds_reader``, no GPU --
which is why it lives beside the pure half of ``extraction`` rather than inside a
task.

An anchor level is a *directory*, never an id token (see ``registry``), because
the failure mode it guards is a silent cross-comparison rather than a crash. That
constraint is what shapes the module: every function here returns the index for
exactly one level, and the level's name is the caller's to pass on.


The cost of an anchor level is the number of anchors, not the number of patients
=================================================================================

This is the whole scoping story for Tier 1.2, and it is worth stating in numbers
because the intuition ("6.7k patients, this is small") is wrong by a factor of 57.

============================  =========  ============================================
level                          anchors    what it buys
============================  =========  ============================================
``lastevent``                     6,731   one vector per patient; the label-free
                                          battery, CKA, and the P1-P4 contrasts
``perlabel-scout``               14,204   the extraction oracle on 9 of 14 tasks
``perlabel``                    381,522   the extraction oracle on all 14 tasks
============================  =========  ============================================

The jump from 14k to 382k is bought by **five tasks**. ``lab_anemia``,
``lab_hyperkalemia``, ``lab_hypoglycemia``, ``lab_hyponatremia`` and
``lab_thrombocytopenia`` label the same ~6k patients at every qualifying lab
draw, so they carry 367,318 of the 381,522 unique anchors -- 96.3% of the
extraction cost for 5/14 of the oracle's evidence. The other nine tasks are one
or two anchors per patient and cost 3.7% between them.

That ratio is why ``perlabel-scout`` exists as its own level. It is not a
sample: it is the *complete* anchor set for nine complete tasks, so each of those
nine is scored on every label it has and compares to its published AUROC without
a subsampling caveat. The five lab tasks are absent rather than thinned, which is
a statement a reader can check, where "we used 4% of the anchors" is one they
cannot.


Why ``perlabel-scout`` is a separate directory and not a filtered ``perlabel``
=============================================================================

Because the alternative silently corrupts the full run. Both levels write
``cells/{cell_id}/embeddings.parquet``; if they shared a directory, running the
scout and later the full set would leave one path whose row count depends on
which job ran last, and every downstream frame would inherit that ambiguity with
nothing in the file to reveal it. Under separate levels, asking for an anchor the
scout does not carry is a missing-row error at the join, which is the behaviour
``registry``'s directory-level rule exists to produce.
"""

import pandas as pd

from embedding_space_exploration.data_management.ehrshot import TASKS, load_labels

# The nine tasks whose anchors are one-or-two-per-patient. Named by exclusion of
# the `lab_` prefix rather than listed, so a task added to `TASKS` lands in the
# right bucket instead of being silently dropped from the scout.
SCOUT_TASKS = tuple(task for task in TASKS if not task.startswith("lab_"))

# The five that carry 96.3% of the anchors. Derived, not asserted, for the same
# reason -- and kept as a name because the split is a costing fact the Slurm
# sizing depends on, not an implementation detail of `SCOUT_TASKS`.
LAB_TASKS = tuple(task for task in TASKS if task.startswith("lab_"))

INDEX_COLUMNS = ("person_id", "cutoff")


def perlabel_index(tasks=TASKS):
    """Every distinct label time in ``tasks``, as an anchor index.

    Deduplicated across tasks: an anchor is a ``(patient, time)`` pair, and two
    tasks labelling one patient at one instant want the same vector, not two
    copies of it. The 14 task files hold 1,152,379 label rows and only 381,522
    distinct anchors between them, so the dedup is a 3x saving before any other
    scoping choice is made.

    Args:
        tasks: Task names to pool. Defaults to all of ``TASKS``; pass
            ``SCOUT_TASKS`` for the nine cheap ones.

    Returns:
        Frame with ``person_id`` and ``cutoff``, one row per distinct anchor.
    """
    labels = pd.concat([load_labels(task) for task in tasks], ignore_index=True)
    anchors = labels.rename(columns={"prediction_time": "cutoff"})
    return (
        anchors[list(INDEX_COLUMNS)]
        .drop_duplicates()
        .sort_values(list(INDEX_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )


def lastevent_index(timeline):
    """One anchor per patient, at the patient's final recorded event.

    The anchor level that needs no blocking decision. ``shared`` -- the level the
    experiment actually reports -- cuts at the last event before a declared
    outcome window, and both the window and the phenotype are open (B1), so
    ``shared`` is not constructible yet. ``lastevent`` ignores the window by
    definition, which is exactly what makes it available now and what makes it a
    robustness arm rather than a replacement.

    The cutoff is the real timestamp rather than ``NaT``. Both keep the whole
    record -- ``events_until`` is inclusive, and the last event is the end of it
    -- so the matrices are identical either way, but a stored ``NaT`` makes every
    row's key indistinguishable from every other patient's, and the join against
    a future ``shared`` index stops being checkable.

    Args:
        timeline: The ``patient_timeline.parquet`` frame, or a path to it.

    Returns:
        Frame with ``person_id`` and ``cutoff``, one row per patient.
    """
    if not isinstance(timeline, pd.DataFrame):
        timeline = pd.read_parquet(timeline)
    anchors = timeline.rename(columns={"last_event": "cutoff"})
    return (
        anchors[list(INDEX_COLUMNS)]
        .sort_values("person_id", kind="stable")
        .reset_index(drop=True)
    )


def build_index(anchor, timeline=None):
    """The anchor index for one level, by name.

    The single place a level name maps to an index, so a Slurm array that passes
    ``--anchor`` and a future pytask that iterates ``registry.ANCHORS`` cannot
    disagree about what a level means.

    Args:
        anchor: One of ``lastevent``, ``perlabel-scout``, ``perlabel``.
        timeline: The timeline frame or path. Required for ``lastevent``.

    Returns:
        Frame with ``person_id`` and ``cutoff``.

    Raises:
        ValueError: For ``shared``, which is gated on B1, and for any unknown
            level.
    """
    if anchor == "lastevent":
        if timeline is None:
            raise ValueError("the lastevent index needs the timeline summary")
        return lastevent_index(timeline)
    if anchor == "perlabel-scout":
        return perlabel_index(SCOUT_TASKS)
    if anchor == "perlabel":
        return perlabel_index(TASKS)
    if anchor == "shared":
        raise ValueError(
            "the shared anchor needs the outcome window declared in Study Design "
            "Freeze section 7, which is gated on decision B1; run 'lastevent' "
            "until that lands"
        )
    raise ValueError(f"unknown anchor level {anchor!r}")
