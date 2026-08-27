"""Tier 1 setup tables, generated from the oracle and timeline artifacts.

Same contract as the Tier 0 tables: every number the prose states is written
from the run that produced it, so the two cannot drift apart.

Two tables. The oracle table is the correctness result -- our probe against the
published EHRSHOT AUROCs. The context table is a property of the cohort rather
than of any model, and it is what decides whether the context-length contrast
has anything to measure.
"""

import pandas as pd
import pytask

from embedding_space_exploration.config import BLD, DOCUMENTS

TABLES = DOCUMENTS / "tables"

# The pretraining context lengths of the 16 Context Clues cells. Transformers
# span 512-4096, the subquadratic architectures 1024-16384.
CONTEXT_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)


@pytask.task(
    kwargs={
        "oracle": BLD / "oracle" / "probe_oracle.parquet",
        "timeline": BLD / "ehrshot" / "patient_timeline.parquet",
        "produces": {
            "oracle": TABLES / "tier1_oracle.md",
            "context": TABLES / "tier1_context.md",
        },
    }
)
def task_tier1_tables(oracle, timeline, produces):
    """Write the oracle comparison and the cohort context table."""
    _write(
        _oracle_table(pd.read_parquet(oracle)),
        produces["oracle"],
        "The Y1 probe against the published EHRSHOT AUROCs, on the CLMBR "
        "features the benchmark itself ships. A task reproduces when the "
        "published value falls inside our bootstrapped 95% CI over 1,000 "
        "resamples of its test split. `new_celiac` carries 94 positives in "
        "7,129 rows, so its interval is too wide to certify anything either "
        "way and it is marked uninformative rather than counted as evidence.",
        "tbl-tier1-oracle",
    )
    _write(
        _context_table(pd.read_parquet(timeline)),
        produces["context"],
        "How much of a patient's record each pretraining context length can "
        "hold, over the 6,731 patients in the extract. Counted in clinical "
        "events, which approximate but do not equal tokens.",
        "tbl-tier1-context",
    )


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _oracle_table(oracle):
    """One row per benchmark task, ours beside the published value."""
    table = oracle.assign(
        ci=lambda d: [
            f"[{low:.3f}, {high:.3f}]"
            for low, high in zip(d.ci_low, d.ci_high, strict=True)
        ],
    )
    return (
        table[
            ["task", "auroc", "ci", "published_auroc", "delta", "n_test", "informative"]
        ]
        .round({"auroc": 3, "published_auroc": 3, "delta": 4})
        .rename(columns={"auroc": "ours", "published_auroc": "published"})
    )


def _context_table(timeline):
    """Share of the cohort whose whole record fits inside each context length."""
    events = timeline["n_events"]
    return pd.DataFrame(
        {
            "context": CONTEXT_LENGTHS,
            "patients_fully_seen": [
                round(100 * (events <= length).mean(), 1) for length in CONTEXT_LENGTHS
            ],
            "median_patient_seen": [
                round(100 * min(length, events.median()) / events.median(), 1)
                for length in CONTEXT_LENGTHS
            ],
        }
    )


def _write(frame, path, caption, label):
    """One MyST table with a caption and a cross-reference label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f":::{{table}} {caption}\n:name: {label}\n\n"
        f"{frame.to_markdown(index=False)}\n:::\n"
    )
