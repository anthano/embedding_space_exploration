"""Materialise the Tier 0 calibration grid: one synthetic space per cell.

Wiring only -- the grid itself, and the reasoning behind every sweep in it, is
declared in ``grid``. Nothing here may be imported to read a constant: see the
note at the top of that module.

Footprint: 138 cells, ~430 MB under ``bld/simulation``, ~20s to build on four
cores.
"""

import pandas as pd
import pytask

from embedding_space_exploration.config import SIMULATION_DIR
from embedding_space_exploration.simulation.generator import simulate_embeddings
from embedding_space_exploration.simulation.grid import FRAMES, GRID

SPEC_FILES = {cell_id: SIMULATION_DIR / cell_id / "spec.parquet" for cell_id in GRID}


for cell_id, cell_kwargs in GRID.items():
    cell_produces = {
        name: SIMULATION_DIR / cell_id / f"{name}.parquet" for name in FRAMES
    }

    @pytask.task(id=cell_id, kwargs={"cell": cell_kwargs, "produces": cell_produces})
    def task_simulate_space(cell, produces):
        """Draw one synthetic space and write its five frames."""
        space = simulate_embeddings(**cell)
        for name, path in produces.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            space[name].to_parquet(path)


@pytask.task(kwargs={"specs": SPEC_FILES, "produces": SIMULATION_DIR / "cells.parquet"})
def task_collect_calibration_grid(specs, produces):
    """Index every cell of the grid in one table, keyed by ``cell_id``.

    What the calibration harness reads to know what it is scoring against, and
    the Tier 0 slice of the selection-budget ledger: one auditable row per
    synthetic configuration actually evaluated, written by the run rather than
    reconstructed at write-up.
    """
    index = pd.concat(
        [
            pd.read_parquet(path).assign(cell_id=cell_id)
            for cell_id, path in specs.items()
        ]
    )
    index.insert(0, "cell_id", index.pop("cell_id"))
    index.sort_values("cell_id").reset_index(drop=True).to_parquet(produces)
