"""The standing correctness oracle: our probe against EHRSHOT's published AUROCs.

This runs the Y1 probe on the CLMBR features **EHRSHOT itself ships** and checks
the result against the AUROCs published for those same features. It therefore
tests exactly one thing: whether our head, splits, label binarisation and scoring
are the protocol that produced the published numbers.

That isolation is the point, and it is why this task exists before any extraction
does. The `allofus` script this replaces embedded patients *and* fitted a head,
then compared one AUROC against the paper -- so a failure could have been a
tokenisation bug, a pooling bug, or a protocol bug, with no way to tell which.
Splitting it gives two independent checks:

- **here** -- protocol, on the release's own vectors, no model required; and
- **Tier 1.2** -- extraction, per cell against the AUROCs Wornow2025 publishes
  for those same cells.

The Tier 1.2 half was originally specified as a *vector-level* check against
these features, which is sharper. It is not runnable: verified 2026-08-27 that
the shipped features come from CLMBR-t-base (JAX/FEMR, vocab 65,536, context
496, rotary) and not from any cell in the grid -- our anchor is a Context Clues
GPT-2 (vocab 39,818, context 512, learned positions). Different embedding tables
cannot produce identical vectors from one timeline. The replacement checks 16
cells we actually run instead of one we do not; see the Build Plan section 1.2.

A failure here means our evaluation is not comparable to the literature, and
every Y1 number downstream is uninterpretable. It is meant to be cheap and to run
on every build, so it stays true rather than being a thing that was true once.
"""

import numpy as np
import pandas as pd
import pytask

from embedding_space_exploration.analysis.probe import fit_probe
from embedding_space_exploration.config import BLD
from embedding_space_exploration.data_management.ehrshot import (
    ASSETS,
    TASKS,
    align_features,
    load_labels,
    load_published_auroc,
    load_reference_features,
    load_splits,
)
from embedding_space_exploration.data_management.splits import split_label

ORACLE_DIR = BLD / "oracle"

# A task reproduces when the published AUROC falls inside our bootstrapped 95%
# CI. Deliberately *not* a fixed absolute tolerance, which the first run showed
# to be wrong in both directions at once: the test-set intervals here span from
# 0.003 (`lab_anemia`, 58k test rows) to 0.28 (`new_celiac`, 94 positives in
# 7,129 rows). One number cannot serve both. A 0.02 gap on `lab_anemia` would be
# a catastrophic failure the tolerance waves through; the same gap on
# `new_celiac` is well inside sampling noise and the tolerance rejects it.
# Comparing against the interval uses the precision each task actually has --
# and section 6 requires us to compute these CIs regardless.

# A CI wider than this certifies little either way, so containment is reported
# but not counted as evidence. `new_celiac` is the case in point: near-chance
# published value, near-chance ours, and an interval wide enough to contain
# both. Its selected C (100, against 0.001-0.01 everywhere else) is the same
# signal read from the other end -- with no signal to fit, the val search picks
# essentially arbitrarily.
INFORMATIVE_CI_WIDTH = 0.20


@pytask.mark.skipif(
    not ASSETS.exists(),
    reason=f"EHRSHOT assets not found at {ASSETS} (set EHRSHOT_ROOT)",
)
@pytask.task(kwargs={"produces": ORACLE_DIR / "probe_oracle.parquet"})
def task_probe_oracle(produces):
    """Score every benchmark task on the shipped features and compare.

    Raises:
        AssertionError: If any task's published AUROC falls outside our
            bootstrapped 95% CI. Loud by design: a silently wrong protocol
            invalidates every Y1 number computed after it.
    """
    index, matrix = load_reference_features()
    splits = load_splits()

    rows = [_score_task(task, index, matrix, splits) for task in TASKS]
    summary = pd.DataFrame(rows)

    produces.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(produces)

    failed = summary[~summary["reproduced"] & summary["published_auroc"].notna()]
    if len(failed):
        raise AssertionError(
            "the published EHRSHOT AUROC falls outside our 95% CI, so our "
            "protocol is not the one behind the published numbers:\n"
            + failed[
                ["task", "auroc", "ci_low", "ci_high", "published_auroc", "delta"]
            ].to_string(index=False)
        )


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def _score_task(task, index, matrix, splits):
    """Probe one task on the reference features and compare to its published row."""
    labels = load_labels(task)
    features, matched = align_features(index, matrix, labels)
    labels = labels[matched].reset_index(drop=True)

    scores = fit_probe(
        features, labels["y"].to_numpy(), split_label(labels["person_id"], splits)
    )
    published = load_published_auroc(task)
    delta = np.nan if published is None else scores["auroc"] - published
    ci_width = scores["ci_high"] - scores["ci_low"]
    return {
        "task": task,
        **scores,
        "published_auroc": published,
        "delta": delta,
        "ci_width": ci_width,
        # NaN comparisons are False throughout, so a task with no published row
        # and one with no scorable split are both counted as demonstrating
        # nothing rather than as a reproduction.
        "reproduced": bool(scores["ci_low"] <= published <= scores["ci_high"])
        if published is not None
        else False,
        "informative": bool(ci_width <= INFORMATIVE_CI_WIDTH),
        "n_labels": len(matched),
        "n_unmatched": int((~matched).sum()),
    }
