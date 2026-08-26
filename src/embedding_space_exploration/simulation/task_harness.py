"""Run the battery over the Tier 0 grid and score it against the planted truth.

Wiring only -- the arms, the gate subset and the reasoning behind both are
declared in ``grid``. Nothing here may be imported to read a constant: see the
note at the top of that module.

Cost: ~280 cell-arms at ~11s, plus ~96 gate runs at ~90s -- on the order of three
hours single-core.
"""

import pandas as pd
import pytask

from embedding_space_exploration.config import CALIBRATION_DIR, SIMULATION_DIR
from embedding_space_exploration.simulation.grid import (
    FRAMES,
    GRID,
    SCALINGS,
    runs_null_gate,
)
from embedding_space_exploration.simulation.harness import score_space


def cell_arm_files(name):
    """One per-cell-arm output across the whole grid, keyed by ``{cell}-{arm}``."""
    return {
        f"{cell_id}-{scaling}": CALIBRATION_DIR / cell_id / scaling / f"{name}.parquet"
        for cell_id in GRID
        for scaling in SCALINGS
    }


for cell_id in GRID:
    for scaling in SCALINGS:
        cell_frames = {
            name: SIMULATION_DIR / cell_id / f"{name}.parquet" for name in FRAMES
        }
        cell_products = {
            "summary": CALIBRATION_DIR / cell_id / scaling / "summary.parquet",
            "curve": CALIBRATION_DIR / cell_id / scaling / "curve.parquet",
        }

        @pytask.task(
            id=f"{cell_id}-{scaling}",
            kwargs={
                "frames": cell_frames,
                "cell_id": cell_id,
                "scaling": scaling,
                "run_null_gate": runs_null_gate(cell_id, scaling),
                "produces": cell_products,
            },
        )
        def task_score_cell(frames, cell_id, scaling, run_null_gate, produces):
            """Score one cell on one preprocessing arm."""
            space = {name: pd.read_parquet(path) for name, path in frames.items()}
            scored = score_space(space, scaling=scaling, run_null_gate=run_null_gate)
            for name, path in produces.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                scored[name].insert(0, "cell_id", cell_id)
                scored[name].to_parquet(path)


@pytask.task(
    kwargs={
        "summaries": {
            f"{cell_id}-{scaling}": CALIBRATION_DIR / cell_id / scaling / f"{n}.parquet"
            for cell_id in GRID
            for scaling in SCALINGS
            for n in ("summary",)
        },
        "curves": {
            f"{cell_id}-{scaling}": CALIBRATION_DIR
            / cell_id
            / scaling
            / "curve.parquet"
            for cell_id in GRID
            for scaling in SCALINGS
        },
        "cells": SIMULATION_DIR / "cells.parquet",
        "produces": {
            "measurements": CALIBRATION_DIR / "measurements.parquet",
            "curves": CALIBRATION_DIR / "curves.parquet",
        },
    }
)
def task_collect_measurements(summaries, curves, cells, produces):
    """Join every measurement to the design that produced it.

    ``measurements.parquet`` is the Tier 0 deliverable: one row per cell-arm,
    carrying every knob that was planted beside every number the battery
    reported. Each sensitivity table in the paper -- and each row of the section
    5 licensing table -- is a group-by on this frame.

    ``curves.parquet`` keeps the per-k detail the summary collapses: the
    prediction-strength curve, the bootstrap-ARI curve, and the null band where
    the gate ran.
    """
    design = pd.read_parquet(cells)
    measurements = pd.concat(
        [pd.read_parquet(path) for path in summaries.values()], ignore_index=True
    )
    measurements = measurements.merge(design, on="cell_id", how="left", validate="m:1")
    measurements.sort_values(["cell_id", "scaling"]).reset_index(drop=True).to_parquet(
        produces["measurements"]
    )

    pd.concat(
        [pd.read_parquet(path) for path in curves.values()], ignore_index=True
    ).sort_values(["cell_id", "scaling", "k"]).reset_index(drop=True).to_parquet(
        produces["curves"]
    )
