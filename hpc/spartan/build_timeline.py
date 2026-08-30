"""Materialise `bld/ehrshot/patient_timeline.parquet` on the cluster.

`bld/` is gitignored, so the timeline summary the `lastevent` anchor is built
from does not arrive with the repo. The pytask that produces it lives in the
root manifest's environment, which is not installed here (see `pixi.toml` on why
this environment is the extraction subset), so this calls the same function
directly.

Run once on a login node, before submitting the array: every array element reads
this file, and having sixteen jobs race to write it is exactly the kind of
corruption that is invisible until a matrix has the wrong number of rows.
"""

import sys

from embedding_space_exploration.config import BLD
from embedding_space_exploration.data_management.timeline import (
    open_database,
    summarise_cohort,
)

OUT = BLD / "ehrshot" / "patient_timeline.parquet"


def main():
    if OUT.exists():
        print(f"{OUT} already exists; nothing to do")
        return 0
    print("summarising the cohort (a few minutes; reads every patient once)...")
    summary = summarise_cohort(open_database())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUT)
    print(f"wrote {OUT} ({len(summary):,} patients)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
