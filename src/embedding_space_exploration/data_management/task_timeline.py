"""Materialise the per-patient timeline summary.

One row per patient in the extract: record bounds, event count, observation
span, age and demographics. Written once and read by everything that needs a
patient fact without needing the timeline itself -- the nuisance-only baseline
(section 1.3), C1's covariate assembler, A3's history lengths, and section 7's
length-stratified reporting.

It is also what settles the section 7 outcome window. That decision needs record
spans, and the numbers available before this task existed were *label* spans,
which are only a lower bound.
"""

import pytask

from embedding_space_exploration.config import BLD
from embedding_space_exploration.data_management.timeline import (
    MEDS_READER_DIR,
    open_database,
    summarise_cohort,
)

EHRSHOT_DIR = BLD / "ehrshot"


@pytask.mark.skipif(
    not MEDS_READER_DIR.exists(),
    reason=f"meds_reader extract not found at {MEDS_READER_DIR} (set EHRSHOT_ROOT)",
)
@pytask.task(kwargs={"produces": EHRSHOT_DIR / "patient_timeline.parquet"})
def task_patient_timeline(produces):
    """Summarise every patient in the extract."""
    summary = summarise_cohort(open_database())
    produces.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(produces)
